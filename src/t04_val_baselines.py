"""T04: validation split, scorer self-test, heuristic baselines, first test submission.

Kaggle usage (after bootstrap):
    import importlib, t04_val_baselines
    importlib.reload(t04_val_baselines)
    t04_val_baselines.main(DATA)
"""
import os
import sys
import time

import polars as pl

from ber.metrics import EMPTY_PAIRS, explode_ids, per_entity_f, self_test, summarize
from ber.normalize import addr_numbers, name_core_tokens
from ber.split import VAL_FOLD, fold_expr


def section(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def load_split(data, split):
    frames = []
    for s in ("1", "2", "3"):
        frames.append(
            pl.read_parquet(f"{data}/{split}_source{s}.parquet",
                            columns=["entity_id", "business_name", "business_address", "country"])
            .with_columns(src=pl.lit("S" + s)))
    return (pl.concat(frames)
              .with_columns(core=name_core_tokens(pl.col("business_name")),
                            num=addr_numbers(pl.col("business_address")),
                            addr_null=pl.col("business_address").is_null())
              .with_columns(k1=pl.col("core").list.join(""),
                            k2=pl.col("core").list.sort().list.join(" "))
              .select("entity_id", "src", "country", "k1", "k2", "num", "addr_null"))


def key_candidates(recs):
    s1 = recs.filter(pl.col("src") == "S1")
    cd = recs.filter(pl.col("src") != "S1")
    parts = []
    for k in ("k1", "k2"):
        a = s1.filter(pl.col(k) != "").select(pl.col("entity_id").alias("s1"), "country", k)
        b = cd.filter(pl.col(k) != "").select(pl.col("entity_id").alias("c"), "country", k)
        parts.append(a.join(b, on=["country", k]).select("s1", "c"))
    cand = pl.concat(parts).unique()
    return (cand.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("num").alias("a_num")), on="s1")
                .join(cd.select(pl.col("entity_id").alias("c"), pl.col("num").alias("b_num"),
                                pl.col("addr_null").alias("b_addr_null")), on="c")
                .with_columns(num_ok=(pl.col("a_num").list.set_intersection(pl.col("b_num"))
                                      .list.len() > 0).fill_null(False))
                .select("s1", "c", "num_ok", "b_addr_null"))


def exclusive(p):
    """Drop candidates claimed by more than one S1 entity (each S2/S3 record has at most one owner)."""
    multi = p.group_by("c").agg(pl.len().alias("m")).filter(pl.col("m") > 1)
    return p.join(multi.select("c"), on="c", how="anti")


def baselines(cand):
    b1 = cand.select("s1", "c")
    return {
        "B0_empty": EMPTY_PAIRS,
        "B1_namekey": b1,
        "B2_namekey_excl": exclusive(b1),
        "B3_namekey_num_excl": exclusive(cand.filter(pl.col("num_ok") | pl.col("b_addr_null"))
                                             .select("s1", "c")),
    }


def write_lists(s1_ids, pairs, col_name, path):
    agg = pairs.group_by("s1").agg(pl.col("c").unique().sort().str.join(","))
    out = (s1_ids.join(agg, on="s1", how="left")
                 .with_columns(pl.col("c").fill_null(""))
                 .rename({"s1": "source1_entity_id", "c": col_name}))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main(data, out_dir="/kaggle/working/output"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()

    section("0. SCORER SELF-TEST")
    ok, got = self_test()
    print(f"self-test {'PASSED' if ok else 'FAILED'}: {[round(x, 4) for x in got]} "
          f"(expected [0.7143, 0.0, 1.0, 0.0])")

    section("1. SPLIT")
    tr = load_split(data, "train")
    gt = explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                     "source1_entity_id", "matched_entity_ids")
    s1 = (tr.filter(pl.col("src") == "S1")
            .select(pl.col("entity_id").alias("s1"), "country", fold_expr()))
    sizes = gt.group_by("s1").agg(pl.len().alias("k"))
    fold_tbl = (s1.join(sizes, on="s1", how="left").with_columns(pl.col("k").fill_null(0))
                  .group_by("fold").agg(pl.len().alias("n"),
                                        (pl.col("country") == "India").mean().alias("india_share"),
                                        (pl.col("k") == 0).mean().alias("singleton_frac"),
                                        pl.col("k").mean().alias("mean_matches"))
                  .sort("fold"))
    print(fold_tbl)
    val = s1.filter(pl.col("fold") == VAL_FOLD).select("s1", "country")
    print(f"validation fold {VAL_FOLD}: {val.height:,} S1 entities")

    section("2. NAME-KEY BLOCKING ON VALIDATION")
    cand = key_candidates(tr)
    vc = cand.join(val, on="s1", how="semi")
    gv = gt.join(val, on="s1", how="semi")
    hit = gv.join(vc.select("s1", "c"), on=["s1", "c"], how="semi")
    print(f"candidate pairs (all train S1): {cand.height:,}")
    print(f"val: {vc.height:,} candidate pairs, {vc.height / val.height:.2f} per S1, "
          f"pair recall ceiling {hit.height / gv.height:.4f}")
    print(vc.join(val, on="s1").group_by("country")
            .agg(pl.len().alias("cand_pairs"), pl.col("num_ok").mean().alias("num_ok_share"),
                 pl.col("b_addr_null").mean().alias("cand_addr_null")).sort("country"))

    section("3. BASELINES ON VALIDATION (macro F0.5)")
    rows = []
    for name, pred in baselines(cand).items():
        e = per_entity_f(pred, gt, val)
        rows.append(summarize(e).with_columns(baseline=pl.lit(name), country=pl.lit("ALL")))
        rows.append(summarize(e, by="country").with_columns(baseline=pl.lit(name)))
    res = pl.concat(rows, how="diagonal_relaxed")
    print(res.select("baseline", "country", "n", "F0.5", "P_avg", "R_avg", "pred_nonempty",
                     "singleton_frac", "F_singletons", "F_nonsingle").sort("baseline", "country"))
    print(f"(train part done in {time.time() - t0:.0f}s)")
    del tr, cand, vc

    section("4. TEST SUBMISSION (B3)")
    te = load_split(data, "test")
    test_s1 = te.filter(pl.col("src") == "S1").select(pl.col("entity_id").alias("s1"), "country")
    cand_t = key_candidates(te)
    pred_t = baselines(cand_t)["B3_namekey_num_excl"]
    os.makedirs(out_dir, exist_ok=True)
    write_lists(test_s1.select("s1"), cand_t.select("s1", "c"), "candidate_entity_ids",
                f"{out_dir}/candidate_pairs.tsv")
    write_lists(test_s1.select("s1"), pred_t, "matched_entity_ids",
                f"{out_dir}/matching_results.tsv")
    summ = (test_s1.join(pred_t.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left")
                   .with_columns(pl.col("k").fill_null(0))
                   .join(cand_t.group_by("s1").agg(pl.len().alias("nc")), on="s1", how="left")
                   .with_columns(pl.col("nc").fill_null(0))
                   .group_by("country").agg(pl.len().alias("s1"),
                                            (pl.col("k") > 0).mean().alias("pred_nonempty"),
                                            pl.col("k").mean().alias("mean_pred"),
                                            pl.col("nc").mean().alias("cands_per_s1"))
                   .sort("country"))
    print(summ)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        p = f"{out_dir}/{f}"
        print(f"wrote {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet",
         sys.argv[2] if len(sys.argv) > 2 else "output")
