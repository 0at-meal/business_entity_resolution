"""T08: learned matcher (LightGBM) over top-K blocked candidates, assignment decisions, test output.

Kaggle usage (after bootstrap; stage01 + stage02 checkpoints in /kaggle/working or attached):
    import t08_matcher
    t08_matcher.main(DATA)
"""
import gc
import glob
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import t04_val_baselines as t04
from ber.decide import decide, decide_expected, tune, tune_expected
from ber.features import FEATS, REC_COLS, context, full_chunked, phase_a
from ber.ids import id_code
from ber.metrics import explode_ids, per_entity_f, summarize
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

TAG = "t16"          # version tag: outputs go to output_{TAG}/, artifacts/*_{TAG}.*
TOPK = 50            # candidates kept per S1 after the cheap pre-filter (T09: 30 -> 50 recovers ~0.8%)
TRAIN_FRAC = 0.35    # share of training-fold S1 entities used to train the model
KEEP_P = 0.01        # test pairs with p below this are discarded before decisions
TOP_SAVE = 10        # per S1, the TOP_SAVE highest-probability pairs are saved for stacking
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=0, seed=7)
ROUNDS = 2500


def find_cand(split, country, work):
    for p in [f"{work}/stage02/cand_{split}_{country}.parquet"] + \
            glob.glob(f"/kaggle/input/**/stage02/cand_{split}_{country}.parquet", recursive=True):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"stage02 candidates for {split}/{country} not found")


def load_cand(split, country, work):
    """Key-blocking candidates (stage02) merged with TF-IDF pass-6 pieces (stage02t), if attached."""
    cand = pl.read_parquet(find_cand(split, country, work))
    hits = (glob.glob(f"{work}/stage02t/tf_{split}_{country}_p*.parquet")
            + glob.glob(f"/kaggle/input/**/stage02t/tf_{split}_{country}_p*.parquet", recursive=True))
    by_name = {}
    for h in hits:
        by_name.setdefault(os.path.basename(h), h)
    if not by_name:
        print(f"    [{split}/{country}] no TF-IDF pieces found - key blocking only", flush=True)
        return cand
    tf = (pl.concat([pl.read_parquet(h) for h in sorted(by_name.values())])
            .unique(subset=["s1", "c"]).with_columns(pl.col("tf_rank").cast(pl.Float32)))
    merged = (cand.join(tf, on=["s1", "c"], how="full", coalesce=True)
                  .with_columns(mask=pl.when(pl.col("tf_cos").is_not_null())
                                     .then(pl.col("mask").fill_null(0).cast(pl.UInt8) | pl.lit(32, pl.UInt8))
                                     .otherwise(pl.col("mask"))))
    added = merged.height - cand.height
    print(f"    [{split}/{country}] TF-IDF pieces {sorted(by_name)}: {tf.height:,} pairs, "
          f"{added:,} new beyond key blocking", flush=True)
    return merged


def countries_of(path):
    return sorted(pl.scan_parquet(path).select("country").unique().collect()["country"].to_list())


def load_recs(path, country):
    return (pl.scan_parquet(path).filter(pl.col("country") == country)
              .select(REC_COLS).collect())


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    os.makedirs(f"{work}/artifacts", exist_ok=True)
    p_tr, p_te = find_checkpoint("train", work), find_checkpoint("test", work)
    print(f"stage01: {p_tr} | {p_te}")

    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    s1_all = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
                .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect())
    val = s1_all.filter(pl.col("fold") == VAL_FOLD).select("s1", "country")
    trn = (s1_all.filter(pl.col("fold") != VAL_FOLD)
                 .sample(fraction=TRAIN_FRAC, seed=11).select("s1", "country"))
    use = pl.concat([trn, val]).select("s1")

    t04.section(f"1. TRAIN FEATURES (top-{TOPK} per S1)")
    parts, ceil_rows = [], []
    for c in countries_of(p_tr):
        t1 = time.time()
        recs = load_recs(p_tr, c)
        cand = load_cand("train", c, work)
        A = context(phase_a(cand, recs, TOPK))
        vc_all = cand.join(val, on="s1", how="semi")
        del cand
        gc.collect()
        gv = gt.join(val.filter(pl.col("country") == c), on="s1", how="semi")
        vA = A.join(val, on="s1", how="semi")
        ceil_rows.append({
            "country": c,
            "recall_before_topk": gv.join(vc_all.select("s1", "c"), on=["s1", "c"], how="semi").height / gv.height,
            "recall_after_topk": gv.join(vA.select("s1", "c"), on=["s1", "c"], how="semi").height / gv.height,
            "per_s1_before": vc_all.height / val.filter(pl.col("country") == c).height,
            "per_s1_after": vA.height / val.filter(pl.col("country") == c).height})
        del vc_all, vA
        F = full_chunked(A.join(use, on="s1", how="semi"), recs)
        parts.append(F)
        print(f"  {c}: {F.height:,} feature rows in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del A, recs
        gc.collect()
    print(pl.DataFrame(ceil_rows))
    D = (pl.concat(parts).join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["s1", "c"], how="left")
           .with_columns(pl.col("y").fill_null(0)))
    del parts
    gc.collect()
    Dtr = D.join(trn, on="s1", how="semi")
    Dv = D.join(val, on="s1", how="semi")
    del D
    print(f"train rows {Dtr.height:,} (pos {Dtr['y'].mean():.3f}); val rows {Dv.height:,}")

    t04.section("2. TRAIN LIGHTGBM")
    dtr = lgb.Dataset(Dtr.select(FEATS).to_numpy(), Dtr["y"].to_numpy(), feature_name=FEATS,
                      free_raw_data=True)
    dv = lgb.Dataset(Dv.select(FEATS).to_numpy(), Dv["y"].to_numpy(), reference=dtr)
    del Dtr
    gc.collect()
    model = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(f"{work}/artifacts/lgbm_{TAG}.txt")
    imp = pl.DataFrame({"feature": FEATS,
                        "gain": model.feature_importance("gain")}).sort("gain", descending=True)
    print(imp.with_columns((pl.col("gain") / pl.col("gain").sum()).alias("share")).head(20))

    t04.section("3. DECISIONS ON VALIDATION")
    pv = Dv.select("s1", "c").with_columns(
        p=pl.Series(model.predict(Dv.select(FEATS).to_numpy(), num_threads=0)).cast(pl.Float32))
    del Dv
    pv.write_parquet(f"{work}/artifacts/val_pred_{TAG}.parquet")
    thr, rel, grid = tune(pv, gt, val)
    print("global threshold rule (top rows):")
    print(grid.head(5).select("thr", "rel", "F0.5", "P_avg", "R_avg", "pred_nonempty", "F_singletons"))
    alpha, floor, grid_e = tune_expected(pv, gt, val)
    print("expected-F rule (top rows):")
    print(grid_e.head(5).select("alpha", "floor", "F0.5", "P_avg", "R_avg", "pred_nonempty", "F_singletons"))
    f_thr = grid.row(0, named=True)["F0.5"]
    f_exp = grid_e.row(0, named=True)["F0.5"]
    use_expected = f_exp > f_thr
    chooser = (lambda df: decide_expected(df, alpha, floor)) if use_expected else (lambda df: decide(df, thr, rel))
    print(f"chosen rule: {'expected-F alpha=%s floor=%s' % (alpha, floor) if use_expected else 'threshold thr=%s rel=%s' % (thr, rel)}")
    e = per_entity_f(chooser(pv), gt, val)
    print(pl.concat([summarize(e).with_columns(country=pl.lit("ALL")), summarize(e, by="country")],
                    how="diagonal_relaxed").select("country", "n", "F0.5", "P_avg", "R_avg",
                                                   "pred_nonempty", "F_singletons", "F_nonsingle"))
    print(f"(T08 reference: 0.9583)  {t04.mem()}")
    del pv, e
    gc.collect()

    t04.section("4. TEST INFERENCE + SUBMISSION")
    ids_te = pl.scan_parquet(p_te).select("id", "entity_id", "src").collect()
    test_s1 = ids_te.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "entity_id")
    idmap = ids_te.filter(pl.col("src") != 1).select("id", "entity_id")
    del ids_te
    cands, preds, trows, tops = [], [], [], []
    for c in countries_of(p_te):
        t1 = time.time()
        recs = load_recs(p_te, c)
        A = context(phase_a(load_cand("test", c, work), recs, TOPK))
        cands.append(A.select("s1", "c"))
        F = full_chunked(A, recs)
        del A, recs
        gc.collect()
        p = F.select("s1", "c").with_columns(
            p=pl.Series(model.predict(F.select(FEATS).to_numpy(), num_threads=0)).cast(pl.Float32))
        preds.append(p.filter(pl.col("p") >= KEEP_P))
        tops.append(p.filter(pl.col("p").rank("ordinal", descending=True).over("s1") <= TOP_SAVE))
        trows.append({"country": c, "pairs": F.height, "kept": preds[-1].height})
        print(f"  {c}: {F.height:,} pairs scored in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del F, p
        gc.collect()
    cand_t = pl.concat(cands)
    all_p = pl.concat(preds)
    all_p.write_parquet(f"{work}/artifacts/test_pred_{TAG}.parquet")
    pl.concat(tops).write_parquet(f"{work}/artifacts/test_top{TOP_SAVE}_{TAG}.parquet")
    pred_t = chooser(all_p)
    out_dir = f"{work}/output_{TAG}"
    os.makedirs(out_dir, exist_ok=True)
    t04.write_lists(test_s1, cand_t, idmap, "candidate_entity_ids", f"{out_dir}/candidate_pairs.tsv")
    t04.write_lists(test_s1, pred_t, idmap, "matched_entity_ids", f"{out_dir}/matching_results.tsv")
    cty = pl.scan_parquet(p_te).filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "country").collect()
    print(pl.DataFrame(trows))
    print(cty.join(pred_t.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left")
             .with_columns(pl.col("k").fill_null(0))
             .group_by("country").agg(pl.len().alias("s1"), (pl.col("k") > 0).mean().alias("pred_nonempty"),
                                      pl.col("k").mean().alias("mean_pred")).sort("country"))
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        print(f"wrote {out_dir}/{f} ({os.path.getsize(f'{out_dir}/{f}') / 1e6:.0f} MB)")
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
