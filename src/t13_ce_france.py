"""T13: adapt the cross-encoder to France with pseudo-labels (GPU).

France has no labels. For each French S1 we take its top-K candidates (quick score) and label them
from our best test predictions: p >= POS -> match, p < NEG -> non-match, in-between -> dropped.
The existing L6 cross-encoder is further trained on these pairs mixed with original US/India pairs
(to avoid forgetting), and monitored on US/India validation pairs.

Needs: stage01 (ber-00), stage02 test candidates (ber-01), ce_model_minilm + ce_data (ber-03),
       test predictions: test_pred_t11 (ber-04, if committed) or test_pred_t10 (ber-01).
Kaggle usage (after bootstrap):
    import t13_ce_france
    t13_ce_france.main(DATA)
"""
import gc
import glob
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import polars as pl
import torch

import t04_val_baselines as t04
from t08_matcher import find_cand, load_recs
from ber.ce import TEXT_COLS, attach_text, train
from ber.features import phase_a
from ber.stage01 import find_checkpoint

BASE = "minilm12"          # model to continue training from (ber-05 output); the T12 blend relies on L12
NAME = "minilm12_fr"       # saved as ce_model_minilm12_fr
COUNTRY = "France"
PRED_TAGS = ["t12", "t11", "t10"]  # first one found is used for pseudo-labels
K = 10
POS, NEG = 0.98, 0.02
MAX_POS = 300_000
NEG_RATIO = 2
ORIG = 600_000             # original US/India pairs mixed in
MAXLEN, BS, LR, EPOCHS = 96, 256, 2e-5, 1


def first_hit(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def main(data, work="/kaggle/working"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pl.Config.set_fmt_str_lengths(70)
    pl.Config.set_tbl_width_chars(250)
    t0 = time.time()
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU"
    p_te = find_checkpoint("test", work)

    pred_path = None
    for tag in PRED_TAGS:
        pred_path = first_hit([f"{work}/artifacts/test_pred_{tag}.parquet",
                               f"/kaggle/input/**/artifacts/test_pred_{tag}.parquet"])
        if pred_path:
            break
    assert pred_path, "no test predictions found: attach ber-04 (t11) or ber-01 (t10) output"
    base_dir = os.path.dirname(first_hit([f"/kaggle/input/**/ce_model_{BASE}/config.json",
                                          f"{work}/ce_model_{BASE}/config.json"]))
    ce_data = os.path.dirname(first_hit(["/kaggle/input/**/ce_data/train_pairs.parquet",
                                         f"{work}/ce_data/train_pairs.parquet"]))
    print(f"pseudo-labels from {pred_path}\nbase model {base_dir}\noriginal pairs {ce_data}")

    t04.section("1. FRENCH PSEUDO-LABELLED PAIRS")
    pt = pl.read_parquet(pred_path).select("s1", "c", "p")
    recs = load_recs(p_te, COUNTRY)
    cand = pl.read_parquet(find_cand("test", COUNTRY, work))
    top = phase_a(cand, recs, K).select("s1", "c")
    del cand, recs
    gc.collect()
    lab = (top.join(pt, on=["s1", "c"], how="left")
              .with_columns(y=pl.when(pl.col("p").is_null() | (pl.col("p") < NEG)).then(0.0)
                             .when(pl.col("p") >= POS).then(1.0).otherwise(None))
              .drop_nulls("y"))
    pos = lab.filter(pl.col("y") == 1.0)
    pos = pos.sample(n=min(MAX_POS, pos.height), seed=5)
    neg = lab.filter(pl.col("y") == 0.0)
    neg = neg.sample(n=min(NEG_RATIO * pos.height, neg.height), seed=5)
    print(f"top-{K} pairs {top.height:,}: confident matches {lab.filter(pl.col('y') == 1.0).height:,}, "
          f"confident non-matches {lab.filter(pl.col('y') == 0.0).height:,}, "
          f"uncertain dropped {top.height - lab.height:,}")
    text = pl.scan_parquet(p_te).filter(pl.col("country") == COUNTRY).select(TEXT_COLS).collect()
    fr = attach_text(pl.concat([pos, neg]).select("s1", "c", "y"), text).select("a_text", "b_text", "y")
    print(f"French training pairs: {fr.height:,} (pos {fr['y'].mean():.3f})")
    print("sample pseudo-labelled matches:")
    print(fr.filter(pl.col("y") == 1.0).sample(6, seed=1))
    print("sample pseudo-labelled non-matches:")
    print(fr.filter(pl.col("y") == 0.0).sample(6, seed=1))

    orig = pl.read_parquet(f"{ce_data}/train_pairs.parquet").select("a_text", "b_text", "y")
    orig = orig.sample(n=min(ORIG, orig.height), seed=5)
    tr = pl.concat([fr, orig]).sample(fraction=1.0, shuffle=True, seed=9)
    va = pl.read_parquet(f"{ce_data}/val_pairs.parquet").select("a_text", "b_text", "y")
    print(f"total training pairs {tr.height:,}; US/India monitoring pairs {va.height:,} "
          f"(original L12 val AUC for reference: 0.99939)")

    t04.section("2. CONTINUE TRAINING")
    tok = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForSequenceClassification.from_pretrained(base_dir)
    d_tr = {"a": tr["a_text"].to_list(), "b": tr["b_text"].to_list(), "y": tr["y"].to_numpy()}
    d_va = {"a": va["a_text"].to_list(), "b": va["b_text"].to_list(), "y": va["y"].to_numpy()}
    del tr, va, orig, fr
    gc.collect()
    best = train(model, tok, d_tr, d_va, maxlen=MAXLEN, bs=BS, lr=LR, epochs=EPOCHS,
                 save_dir=f"{work}/ce_model_{NAME}")
    print(f"\nbest US/India val AUC {best:.5f}; saved {work}/ce_model_{NAME}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
