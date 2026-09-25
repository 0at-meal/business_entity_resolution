"""T09: error analysis of the T08 matcher on the validation fold (no retraining).

Needs: stage01 (ber-00 output), stage02 candidates + artifacts/lgbm_t08.txt (ber-01 output).
Kaggle usage (after bootstrap):
    import t09_errors
    t09_errors.main(DATA)
"""
import gc
import glob
import os
import sys
import time

import lightgbm as lgb
import polars as pl

import t04_val_baselines as t04
from t06b_normalize import raw_for_ids
from t08_matcher import TOPK, countries_of, find_cand, load_recs
from ber.decide import decide
from ber.features import FEATS, context, full_chunked, phase_a
from ber.ids import id_code
from ber.metrics import explode_ids, per_entity_f, summarize
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

THR = 0.7
K_MAX = 80
DIAG = ["q", "n_tsort", "a_tsr", "num_inter", "state_rel", "q_rank_c", "q_gap_c", "n_cand_c",
        "b_nonlatin", "b_addr_null"]


def find_model(work):
    for p in [f"{work}/artifacts/lgbm_t08.txt"] + \
            glob.glob("/kaggle/input/**/artifacts/lgbm_t08.txt", recursive=True):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("lgbm_t08.txt not found: attach the ber-01 notebook output")


def show(data, df, title, n=8):
    print(f"\n--- {title}: {df.height:,} pairs (showing {min(n, df.height)}) ---")
    if df.height == 0:
        return
    sub = df.sample(min(n, df.height), seed=1)
    raw = raw_for_ids(data, "train", sub["s1"].to_list() + sub["c"].to_list()).select(
        "id", nm=pl.col("business_name").str.slice(0, 38), ad=pl.col("business_address").str.slice(0, 48))
    out = (sub.join(raw.rename({"id": "s1", "nm": "s1_name", "ad": "s1_addr"}), on="s1", how="left")
              .join(raw.rename({"id": "c", "nm": "c_name", "ad": "c_addr"}), on="c", how="left"))
    cols = ["s1_name", "c_name", "s1_addr", "c_addr"] + [c for c in ["p", "q", "n_tsort", "a_tsr",
                                                                     "num_inter", "state_rel", "q_rank_c"]
                                                         if c in out.columns]
    print(out.select(cols))


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(400)
    pl.Config.set_fmt_str_lengths(50)
    pl.Config.set_float_precision(3)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    p_tr = find_checkpoint("train", work)
    model = lgb.Booster(model_file=find_model(work))
    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    s1_all = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
                .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect())
    val = s1_all.filter(pl.col("fold") == VAL_FOLD).select("s1", "country")

    t04.section("1. RE-SCORE VALIDATION WITH THE SAVED MODEL")
    pv_parts, cand_parts, k_rows = [], [], []
    for c in countries_of(p_tr):
        t1 = time.time()
        recs = load_recs(p_tr, c)
        cand = pl.read_parquet(find_cand("train", c, work))
        val_c = val.filter(pl.col("country") == c).select("s1")
        cand_parts.append(cand.join(val_c, on="s1", how="semi"))
        A = phase_a(cand, recs, K_MAX)
        del cand
        gc.collect()
        A = A.with_columns(r=pl.col("q").rank("ordinal", descending=True).over("s1"))
        gv = gt.join(val_c, on="s1", how="semi")
        vA = A.join(val_c, on="s1", how="semi")
        for k in (TOPK, 50, K_MAX):
            k_rows.append({"country": c, "K": k,
                           "recall_after_topk": gv.join(vA.filter(pl.col("r") <= k).select("s1", "c"),
                                                        on=["s1", "c"], how="semi").height / gv.height,
                           "pairs_per_s1": vA.filter(pl.col("r") <= k).height / val_c.height})
        A = context(A.filter(pl.col("r") <= TOPK).drop("r"))
        V = full_chunked(A.join(val_c, on="s1", how="semi"), recs)
        del A, recs, vA
        gc.collect()
        pv_parts.append(V.select("s1", "c", *DIAG).with_columns(
            p=pl.Series(model.predict(V.select(FEATS).to_numpy(), num_threads=0)).cast(pl.Float32)))
        print(f"  {c}: {V.height:,} val pairs scored in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del V
        gc.collect()
    pv = pl.concat(pv_parts)
    cand_v = pl.concat(cand_parts)
    del pv_parts, cand_parts
    print(pl.DataFrame(k_rows))
    pred = decide(pv.select("s1", "c", "p"), THR)
    e = per_entity_f(pred, gt, val)
    print(summarize(e).select("n", "F0.5", "P_avg", "R_avg", "pred_nonempty", "F_singletons"))
    print("(T08 reference: F0.5 0.9583)")

    t04.section("2. WHERE MISSED TRUE PAIRS WERE LOST")
    gv = gt.join(val, on="s1", how="semi")
    fn = (gv.join(val, on="s1")
            .join(cand_v.select("s1", "c", pl.lit(True).alias("in_cand")), on=["s1", "c"], how="left")
            .join(pv, on=["s1", "c"], how="left")
            .join(pred.with_columns(pred=pl.lit(True)), on=["s1", "c"], how="left")
            .with_columns(cat=pl.when(pl.col("in_cand").is_null()).then(pl.lit("not_blocked"))
                               .when(pl.col("p").is_null()).then(pl.lit("cut_by_topk"))
                               .when(pl.col("p") < THR).then(pl.lit("below_thr"))
                               .when(pl.col("pred").is_null()).then(pl.lit("lost_assignment"))
                               .otherwise(pl.lit("found"))))
    print(fn.group_by("country", "cat").agg(pl.len().alias("n"))
            .with_columns(share=pl.col("n") / pl.col("n").sum().over("country"))
            .sort("country", "n", descending=[False, True]))
    bt = fn.filter(pl.col("cat") == "below_thr")
    if bt.height:
        print("below_thr probability quantiles: " +
              "  ".join(f"p{int(q * 100)}={bt['p'].quantile(q):.3f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
    nothing = (e.filter((pl.col("n_true") > 0) & (pl.col("n_pred") == 0)).select("s1"))
    print(f"non-singleton S1 with NO prediction: {nothing.height:,} "
          f"({nothing.height / e.filter(pl.col('n_true') > 0).height:.4f}); their pairs by category:")
    print(fn.join(nothing, on="s1", how="semi").group_by("cat").agg(pl.len().alias("n")).sort("n", descending=True))
    print("missed pairs by candidate type (found vs not):")
    print(fn.with_columns(found=pl.col("cat") == "found")
            .group_by("b_nonlatin", "b_addr_null").agg(pl.len().alias("n"), pl.col("found").mean().alias("found_rate"))
            .sort("n", descending=True))

    t04.section("3. WHAT THE FALSE MATCHES ARE")
    owners = gt.select("c", pl.col("s1").alias("true_owner"))
    singles = e.filter(pl.col("n_true") == 0).select("s1")
    fp = (pred.join(gv, on=["s1", "c"], how="anti")
              .join(val, on="s1")
              .join(owners, on="c", how="left")
              .join(singles.with_columns(single=pl.lit(True)), on="s1", how="left")
              .join(pv, on=["s1", "c"], how="left")
              .with_columns(cat=pl.when(pl.col("single").is_not_null()).then(pl.lit("on_singleton"))
                                 .when(pl.col("true_owner").is_not_null()).then(pl.lit("owned_by_other_s1"))
                                 .otherwise(pl.lit("record_owned_by_nobody"))))
    print(f"false matches: {fp.height:,} of {pred.height:,} predicted pairs")
    print(fp.group_by("country", "cat").agg(pl.len().alias("n")).sort("country", "n", descending=[False, True]))

    t04.section("4. EXAMPLES")
    for cat in ["not_blocked", "cut_by_topk", "below_thr", "lost_assignment"]:
        show(data, fn.filter(pl.col("cat") == cat), f"MISSED / {cat}")
    for cat in ["owned_by_other_s1", "on_singleton", "record_owned_by_nobody"]:
        show(data, fp.filter(pl.col("cat") == cat), f"FALSE MATCH / {cat}")
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
