"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

Stages (each writes checkpoints under WORK so later stages can be rerun alone):

  convert   TSV -> Parquet                                   data/parquet/*.parquet          CPU  ~2 min
  normalize names/addresses, native-script dictionary        WORK/stage01/{train,test}        CPU  ~20 min
  block     key-based blocking, 5 passes                     WORK/stage02/cand_*              CPU  ~15 min
  tfidf     char 3-gram TF-IDF nearest neighbours (pass 6)   WORK/stage02t/tf_*               CPU  ~1 h per shard
  matcher   LightGBM on top-50 + TF-IDF candidates           WORK/artifacts/*_t16, output_t16 CPU  ~5.5 h
  ce        cross-encoders (+ French adaptation)             WORK/ce_model_*                  GPU  ~1-3 h each
  blend     cross-encoder rescoring + logistic blend         WORK/output_<tag>/               GPU  ~1 h
  stack     (alternative final) cross-encoders on every top-10 pair + 2nd-stage LightGBM
            = scoring shards (GPU ~1.5 h each) + stacking (CPU ~10 min) -> WORK/output_t19/

Usage:
    python src/run_pipeline.py --stage all   --raw data/raw --data data/parquet --work work
    python src/run_pipeline.py --stage blend --data data/parquet --work work

The TF-IDF stage is split into shards (train/test x country x part) because the search is slow;
on Kaggle we ran the 7 shards in parallel notebooks. Here they run one after another.
The 'ce' and 'blend' stages need a CUDA GPU (we used 2 x Tesla T4).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ----------------------------------------------------------------------------- configuration
TFIDF_SHARDS = [  # (split, country, part, n_parts)
    ("train", "India", 0, 1), ("train", "US", 0, 1),
    ("test", "India", 0, 3), ("test", "India", 1, 3), ("test", "India", 2, 3),
    ("test", "US", 0, 1), ("test", "France", 0, 1),
]
CROSS_ENCODERS = [  # arguments of t11a_ce_train.main (MiniLM: Apache-2.0, DeBERTa-v3: MIT)
    dict(model="cross-encoder/ms-marco-MiniLM-L-6-v2", name="minilm"),
    dict(model="cross-encoder/ms-marco-MiniLM-L-12-v2", name="minilm12"),
    dict(model="cross-encoder/ms-marco-MiniLM-L-12-v2", name="minilm12v2",
         frac=0.3, epochs=2, bs=512, lr=4e-5),
    dict(model="microsoft/deberta-v3-small", name="debertav3s", frac=0.15, epochs=1, bs=256, lr=3e-5),
]
FRENCH_ADAPTATION = [  # arguments of t13_ce_france.main (pseudo-labels from T16 test predictions)
    dict(base="minilm12v2", name="minilm12v2_fr", pred_tags=["t16"]),
]
# FINAL: which last stage produces the submission ("blend" = T17b, "stack" = T19).
FINAL = "stack"   # T19 scored best on the leaderboard (0.978)
# T17b: cross-encoder rescoring of uncertain pairs + logistic blend, no French override.
FINAL_BLEND = dict(src_tag="t16", tag="t17b",
                   ce_names=["minilm", "minilm12", "minilm12v2"],
                   fr_override={})
# T19: cross-encoders on every top-10 pair (+ French L12 v2 for French pairs) + 2nd-stage model.
STACK_CE = ["minilm12v2", "debertav3s"]
STACK_SHARDS = [("val", 0, 1), ("test", 0, 3), ("test", 1, 3), ("test", 2, 3)]
STAGES = ["convert", "normalize", "block", "tfidf", "matcher", "ce", FINAL]


def run(stage, a):
    print(f"\n######## stage: {stage} ########", flush=True)
    if stage == "convert":
        subprocess.run([sys.executable, os.path.join(HERE, "stage00_convert_profile.py"),
                        "--raw-dir", a.raw, "--out-dir", a.data,
                        "--report", os.path.join(a.work, "reports", "profile.json")], check=True)
    elif stage == "normalize":
        import t06b_normalize
        t06b_normalize.main(a.data, a.work)
    elif stage == "block":
        import t07_blocking
        t07_blocking.main(a.data, a.work)
    elif stage == "tfidf":
        import t15b_tfidf_block
        for split, country, part, n in TFIDF_SHARDS:
            t15b_tfidf_block.main(a.data, split, country, part=part, n_parts=n, work=a.work)
    elif stage == "matcher":
        import t08_matcher
        t08_matcher.main(a.data, a.work)
    elif stage == "ce":
        import t11a_ce_train
        import t13_ce_france
        for kw in CROSS_ENCODERS:
            t11a_ce_train.main(a.data, a.work, **kw)
        for kw in FRENCH_ADAPTATION:
            t13_ce_france.main(a.data, a.work, **kw)
    elif stage in ("blend", "stack"):
        import shutil
        if stage == "blend":
            import t11b_ce_blend
            t11b_ce_blend.main(a.data, a.work, **FINAL_BLEND)
            out = os.path.join(a.work, f"output_{FINAL_BLEND['tag']}")
        else:
            import t19_stack
            for split, part, n in STACK_SHARDS:
                t19_stack.score(a.data, split, part, n, work=a.work, ce_names=STACK_CE)
            t19_stack.stack(a.data, a.work)
            out = os.path.join(a.work, "output_t19")
        os.makedirs(a.out, exist_ok=True)
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy(os.path.join(out, f), os.path.join(a.out, f))
        print(f"final files copied to {a.out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "convert", "normalize", "block", "tfidf",
                                                       "matcher", "ce", "blend", "stack"])
    ap.add_argument("--raw", default="data/raw", help="folder containing train/ and test/ TSVs")
    ap.add_argument("--data", default="data/parquet", help="Parquet folder (written by 'convert')")
    ap.add_argument("--work", default="work", help="checkpoints, models and intermediate outputs")
    ap.add_argument("--out", default="output", help="where the two submission TSVs are copied")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for stage in (STAGES if a.stage == "all" else [a.stage]):
        run(stage, a)


if __name__ == "__main__":
    main()
