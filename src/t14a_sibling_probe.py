"""T14a: can confidently matched siblings reach true pairs that normal blocking missed?

For each validation S1, confident matches (T10 validation probability >= CONF, one owner per
record) act as "siblings". The five blocking passes are run FROM each sibling against all S2/S3
records. Any record reached this way becomes a sibling-connected candidate of that S1.

Reports per country:
  * not_blocked            true pairs normal blocking never generated
  * reachable              ... of those, reached through a sibling (the prize)
  * new_per_s1 / new_true  candidate volume the expansion adds, and how many are true
  * doubtful-pair signal   true-match rate of uncertain candidates with vs without a sibling link

Needs: stage01 (ber-00), stage02 v3 candidates + artifacts/val_pred_t10.parquet (latest ber-01).
Kaggle usage (after bootstrap):
    import t14a_sibling_probe
    t14a_sibling_probe.main(DATA)
"""
import gc
import glob
import sys
import time

import polars as pl

import t04_val_baselines as t04
from t08_matcher import countries_of, find_cand
from ber.blocking import PASSES, _keys, _pairs, prepare
from ber.decide import decide
from ber.ids import id_code
from ber.metrics import explode_ids
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

CONF = 0.9
DOUBT = (0.02, 0.7)
COLS = ["id", "src", "country", "k1", "k2", "name_core", "num", "addr_toks", "state"]


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()

    p_tr = find_checkpoint("train", work)
    hits = (glob.glob(f"{work}/artifacts/val_pred_t10.parquet")
            + glob.glob("/kaggle/input/**/artifacts/val_pred_t10.parquet", recursive=True))
    assert hits, "val_pred_t10.parquet not found: attach the LATEST ber-01 version (T10 run)"
    pv = pl.read_parquet(hits[0]).select("s1", "c", "p")
    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    val = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
             .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect()
             .filter(pl.col("fold") == VAL_FOLD).select("s1", "country"))
    M_all = decide(pv, CONF).rename({"c": "m"})
    print(f"confident sibling links (p >= {CONF}): {M_all.height:,} "
          f"({M_all.height / val.height:.2f} per val S1), precision "
          f"{M_all.join(gt.rename({'c': 'm'}), on=['s1', 'm'], how='semi').height / M_all.height:.4f}")

    rows, sig_rows = [], []
    for c in countries_of(p_tr):
        t1 = time.time()
        val_c = val.filter(pl.col("country") == c).select("s1")
        df = prepare(pl.scan_parquet(p_tr).filter(pl.col("country") == c).select(COLS).collect())
        src = df.select("id", "src")
        M = M_all.join(val_c, on="s1", how="semi")
        mids = M.select(pl.col("m").alias("id")).unique()
        cand = pl.read_parquet(find_cand("train", c, work)).join(val_c, on="s1", how="semi").select("s1", "c")
        gv = gt.join(val_c, on="s1", how="semi")
        nb = gv.join(cand, on=["s1", "c"], how="anti")

        parts = []
        for kind, cap in PASSES:
            keys = _keys(df, kind).join(src, on="id")
            kq = keys.join(mids, on="id", how="semi").select("id", "key")
            kc = keys.filter(pl.col("src") != 1).select("id", "key")
            p, _ = _pairs(kq, kc, cap)
            parts.append(p.rename({"s1": "m", "c": "r"}))
            del keys, kq, kc
            gc.collect()
        mr = pl.concat(parts).unique().filter(pl.col("m") != pl.col("r"))
        del parts, df
        sr = (M.join(mr, on="m").select("s1", pl.col("r").alias("c")).unique()
               .join(M.rename({"m": "c"}), on=["s1", "c"], how="anti"))   # drop the siblings themselves
        new = sr.join(cand, on=["s1", "c"], how="anti")
        reach = nb.join(new, on=["s1", "c"], how="semi")
        found_before = gv.height - nb.height
        rows.append({
            "country": c,
            "val_true_pairs": gv.height,
            "not_blocked": nb.height,
            "nb_share": nb.height / gv.height,
            "nb_s1_has_sibling": nb.join(M.select("s1").unique(), on="s1", how="semi").height / max(nb.height, 1),
            "reachable": reach.height,
            "reach_share_of_nb": reach.height / max(nb.height, 1),
            "new_per_s1": new.height / val_c.height,
            "new_true_rate": new.join(gv, on=["s1", "c"], how="semi").height / max(new.height, 1),
            "ceiling_now": found_before / gv.height,
            "ceiling_with_siblings": (found_before + reach.height) / gv.height,
        })
        d = (pv.join(val_c, on="s1", how="semi")
               .filter(pl.col("p").is_between(*DOUBT))
               .join(gt.with_columns(y=pl.lit(1)), on=["s1", "c"], how="left")
               .with_columns(pl.col("y").fill_null(0))
               .join(sr.with_columns(sib=pl.lit(True)), on=["s1", "c"], how="left")
               .with_columns(pl.col("sib").fill_null(False)))
        for flag in (True, False):
            dd = d.filter(pl.col("sib") == flag)
            sig_rows.append({"country": c, "sibling_link": flag, "doubtful_pairs": dd.height,
                             "true_rate": dd["y"].mean() if dd.height else None})
        print(f"  {c}: done in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        del mr, sr, new, cand, M
        gc.collect()

    t04.section("1. BLOCKING: WHAT SIBLINGS COULD ADD (validation)")
    res = pl.DataFrame(rows)
    print(res)
    tot = res.select(pl.col("val_true_pairs").sum(), pl.col("not_blocked").sum(), pl.col("reachable").sum())
    vt, nbt, rt = tot.row(0)
    print(f"ALL: not blocked {nbt:,} ({nbt / vt:.4f} of true pairs); reachable via siblings {rt:,} "
          f"({rt / max(nbt, 1):.3f} of those); ceiling {1 - nbt / vt:.4f} -> {(vt - nbt + rt) / vt:.4f}")

    t04.section("2. DOUBTFUL CANDIDATES: DOES A SIBLING LINK SIGNAL A TRUE MATCH?")
    print(pl.DataFrame(sig_rows))
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
