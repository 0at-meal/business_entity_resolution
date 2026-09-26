"""T11a: fine-tune a cross-encoder reranker on hard candidate pairs (GPU).

Needs: stage01 (ber-00 output) and stage02 candidates (ber-01 output), GPU + internet.
Kaggle usage (after bootstrap):
    import t11a_ce_train
    t11a_ce_train.main(DATA)
"""
import gc
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import polars as pl
import torch

import t04_val_baselines as t04
from t08_matcher import countries_of, find_cand, load_recs
from ber.ce import TEXT_COLS, attach_text, train
from ber.features import phase_a
from ber.ids import id_code
from ber.metrics import explode_ids
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # Apache-2.0; swap for a larger model later
NAME = "minilm"
TRAIN_S1_FRAC = 0.15    # of training-fold S1 entities
VAL_S1 = 20_000         # validation S1 entities used for monitoring
K = 10                  # hardest candidates per S1 (by quick score) + all true matches
MAXLEN = 96
BS = 256
LR = 3e-5
EPOCHS = 1


def build_pairs(p_tr, gt, s1_sel, work):
    """Top-K candidates by quick score for the selected S1s, plus all their true matches."""
    parts = []
    for c in countries_of(p_tr):
        t1 = time.time()
        recs = load_recs(p_tr, c)
        sel = s1_sel.filter(pl.col("country") == c).select("s1")
        cand = pl.read_parquet(find_cand("train", c, work)).join(sel, on="s1", how="semi")
        top = phase_a(cand, recs, K).select("s1", "c")
        pos = gt.join(sel, on="s1", how="semi").join(cand.select("s1", "c"), on=["s1", "c"], how="semi")
        pairs = pl.concat([top, pos]).unique()
        text = pl.scan_parquet(p_tr).filter(pl.col("country") == c).select(TEXT_COLS).collect()
        parts.append(attach_text(pairs, text))
        print(f"  {c}: {pairs.height:,} pairs in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del recs, cand, top, pos, text
        gc.collect()
    df = pl.concat(parts).join(gt.with_columns(y=pl.lit(1.0)), on=["s1", "c"], how="left")
    return df.with_columns(pl.col("y").fill_null(0.0)).sample(fraction=1.0, shuffle=True, seed=3)


def main(data, work="/kaggle/working", model=None, name=None, frac=None, epochs=None):
    """model/name/frac/epochs override the module defaults (e.g. a larger cross-encoder)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    global MODEL, NAME, TRAIN_S1_FRAC, EPOCHS
    MODEL = model or MODEL
    NAME = name or NAME
    TRAIN_S1_FRAC = frac or TRAIN_S1_FRAC
    EPOCHS = epochs or EPOCHS

    t0 = time.time()
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU in the notebook settings"
    print(f"GPU: {torch.cuda.get_device_name(0)} | model {MODEL}")
    os.makedirs(f"{work}/ce_data", exist_ok=True)
    p_tr = find_checkpoint("train", work)
    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    s1_all = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
                .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect())
    trn = s1_all.filter(pl.col("fold") != VAL_FOLD).sample(fraction=TRAIN_S1_FRAC, seed=21)
    val = s1_all.filter(pl.col("fold") == VAL_FOLD).sample(n=VAL_S1, seed=21)

    t04.section("1. BUILD PAIR TEXT")
    tr = build_pairs(p_tr, gt, trn, work)
    va = build_pairs(p_tr, gt, val, work)
    tr.write_parquet(f"{work}/ce_data/train_pairs.parquet")
    va.write_parquet(f"{work}/ce_data/val_pairs.parquet")
    print(f"train pairs {tr.height:,} (pos {tr['y'].mean():.3f}); val pairs {va.height:,} (pos {va['y'].mean():.3f})")
    print(tr.select("a_text", "b_text", "y").head(6))

    t04.section("2. FINE-TUNE")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    d_tr = {"a": tr["a_text"].to_list(), "b": tr["b_text"].to_list(), "y": tr["y"].to_numpy()}
    d_va = {"a": va["a_text"].to_list(), "b": va["b_text"].to_list(), "y": va["y"].to_numpy()}
    del tr, va
    gc.collect()
    best = train(model, tok, d_tr, d_va, maxlen=MAXLEN, bs=BS, lr=LR, epochs=EPOCHS,
                 save_dir=f"{work}/ce_model_{NAME}")
    print(f"\nbest val AUC {best:.5f}; model saved to {work}/ce_model_{NAME}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
