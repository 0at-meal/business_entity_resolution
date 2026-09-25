"""T07: multi-pass blocking, measured on validation; saves stage-02 candidate checkpoints.

Kaggle usage (after bootstrap; stage01 checkpoint attached or in /kaggle/working):
    import t07_blocking
    t07_blocking.main(DATA)
"""
import gc
import os
import sys
import time

import polars as pl

import t04_val_baselines as t04
from ber.blocking import PASSES, generate
from ber.ids import id_code
from ber.metrics import explode_ids
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

COLS = ["id", "src", "country", "k1", "k2", "name_core", "num", "addr_toks"]


def eval_country(cand, gv, val_ids, country):
    """Per-pass alone / cumulative recall, pairs per S1 and coverage on validation."""
    vc = cand.join(val_ids, on="s1", how="semi")
    n_val = val_ids.height
    nonsingle = gv.select("s1").unique()
    rows = []
    for i, (kind, _) in enumerate(PASSES):
        bit, cum = 1 << i, (1 << (i + 1)) - 1
        alone = vc.filter((pl.col("mask") & bit) > 0)
        upto = vc.filter((pl.col("mask") & cum) > 0)
        hit_a = gv.join(alone.select("s1", "c"), on=["s1", "c"], how="semi")
        hit_c = gv.join(upto.select("s1", "c"), on=["s1", "c"], how="semi")
        rows.append({
            "country": country, "pass": kind,
            "alone_recall": hit_a.height / max(gv.height, 1),
            "alone_per_s1": alone.height / n_val,
            "cum_recall": hit_c.height / max(gv.height, 1),
            "cum_per_s1": upto.height / n_val,
            "cum_coverage": hit_c.select("s1").n_unique() / max(nonsingle.height, 1),
        })
    zero = val_ids.join(vc.select("s1").unique(), on="s1", how="anti").height / n_val
    return rows, {"country": country, "val_s1": n_val, "val_true_pairs": gv.height,
                  "zero_cand_share": zero, "cands_p50": None}


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    os.makedirs(f"{work}/stage02", exist_ok=True)

    p_tr, p_te = find_checkpoint("train", work), find_checkpoint("test", work)
    print(f"stage01 train: {p_tr}\nstage01 test:  {p_te}")
    if not p_tr or not p_te:
        raise FileNotFoundError("stage01 checkpoint not found: attach the T06b notebook output or rerun T06b")

    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    s1_all = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
                .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect())
    val = s1_all.filter(pl.col("fold") == VAL_FOLD).select("s1", "country")

    t04.section("1. TRAIN BLOCKING (per country)")
    all_rows, summ, stats_all = [], [], []
    countries = sorted(pl.scan_parquet(p_tr).select("country").unique().collect()["country"].to_list())
    for c in countries:
        t1 = time.time()
        df = pl.scan_parquet(p_tr).filter(pl.col("country") == c).select(COLS).collect()
        cand, stats = generate(df)
        del df
        gc.collect()
        cand.write_parquet(f"{work}/stage02/cand_train_{c}.parquet", compression="zstd")
        val_c = val.filter(pl.col("country") == c).select("s1")
        gv = gt.join(val_c, on="s1", how="semi")
        rows, s = eval_country(cand, gv, val_c, c)
        s["all_pairs"] = cand.height
        s["pairs_per_s1_all"] = cand.height / s1_all.filter(pl.col("country") == c).height
        all_rows += rows
        summ.append(s)
        stats_all += [{"split": "train", "country": c, **x} for x in stats]
        print(f"  {c}: {cand.height:,} candidate pairs in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del cand
        gc.collect()

    t04.section("2. VALIDATION RECALL BY PASS (alone and cumulative, in pass order)")
    res = pl.DataFrame(all_rows)
    print(res)
    tot = []
    for i, (kind, _) in enumerate(PASSES):
        r = res.filter(pl.col("pass") == kind)
        w = pl.DataFrame(summ).select("country", "val_s1", "val_true_pairs")
        j = r.join(w, on="country")
        tot.append({"pass": kind,
                    "cum_recall_ALL": (j["cum_recall"] * j["val_true_pairs"]).sum() / j["val_true_pairs"].sum(),
                    "cum_per_s1_ALL": (j["cum_per_s1"] * j["val_s1"]).sum() / j["val_s1"].sum(),
                    "cum_coverage_ALL": (j["cum_coverage"] * j["val_s1"]).sum() / j["val_s1"].sum()})
    print(pl.DataFrame(tot))
    print(pl.DataFrame(summ).drop("cands_p50"))
    print("reference: name keys alone (T06b) ceiling 0.6463, 39.3 candidates per S1")

    t04.section("3. TEST BLOCKING (per country)")
    tcountries = sorted(pl.scan_parquet(p_te).select("country").unique().collect()["country"].to_list())
    trows = []
    for c in tcountries:
        t1 = time.time()
        df = pl.scan_parquet(p_te).filter(pl.col("country") == c).select(COLS).collect()
        n_s1 = df.filter(pl.col("src") == 1).height
        cand, stats = generate(df)
        del df
        gc.collect()
        cand.write_parquet(f"{work}/stage02/cand_test_{c}.parquet", compression="zstd")
        per = cand.group_by("s1").agg(pl.len().alias("n"))
        trows.append({"country": c, "s1": n_s1, "pairs": cand.height, "per_s1": cand.height / n_s1,
                      "zero_cand_share": 1 - per.height / n_s1,
                      "per_s1_p50": per["n"].median(), "per_s1_p99": per["n"].quantile(0.99)})
        stats_all += [{"split": "test", "country": c, **x} for x in stats]
        print(f"  {c}: {cand.height:,} candidate pairs in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del cand, per
        gc.collect()
    print(pl.DataFrame(trows))

    t04.section("4. PASS SIZES (all S1, before union)")
    print(pl.DataFrame(stats_all))
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
