"""T05: unlabeled test-set inspection, focused on France (unseen in training).

Uses the T04 B3 test predictions (~96% precision on validation) as pseudo-labels to show
how French records vary across sources.

Kaggle usage (after bootstrap, same session as T04 so the submission file exists):
    import importlib, t05_test_inspect
    importlib.reload(t05_test_inspect)
    t05_test_inspect.main(DATA)
"""
import os
import sys
import time

import polars as pl

import t03_eda_pairs as t03
from ber.metrics import explode_ids


def load_test(data):
    frames = []
    for s in ("1", "2", "3"):
        frames.append(
            pl.read_parquet(f"{data}/test_source{s}.parquet",
                            columns=["entity_id", "business_name", "business_address", "country"])
            .with_columns(src=pl.lit("S" + s)))
    return pl.concat(frames)


def main(data, sub_path="/kaggle/working/output/matching_results.tsv"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(400)
    pl.Config.set_fmt_str_lengths(80)
    pl.Config.set_float_precision(3)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()

    recs = load_test(data)
    print(f"loaded {recs.height:,} test records in {time.time() - t0:.0f}s")

    t03.section("1. TEST SCRIPTS / ACCENTS / NULLS (share of records)")
    t03.sec_scripts(recs)

    t03.section("2. FRANCE NAME TAIL TOKENS / ADDRESS TOKENS (sampled)")
    fr = (recs.filter(pl.col("country") == "France")
              .with_columns(nn=t03.norm(pl.col("business_name")),
                            na=t03.norm(pl.col("business_address")))
              .with_columns(nt=pl.col("nn").str.split(" "),
                            at=pl.col("na").str.split(" ")))
    t03.sec_tokens(fr)

    t03.section("2b. FRANCE FIRST NAME TOKENS (all sources, sampled)")
    sub = fr.sample(min(1_000_000, fr.height), seed=0)
    print(t03._top(sub.select(t=pl.col("nt").list.first()), 60))

    t03.section("3. FRANCE PSEUDO-CLUSTERS FROM B3 PREDICTIONS")
    if not os.path.exists(sub_path):
        print(f"{sub_path} not found - rerun T04 in this session first, then rerun T05.")
    else:
        subm = pl.read_csv(sub_path, separator="\t", quote_char=None, infer_schema_length=0)
        pairs = explode_ids(subm, "source1_entity_id", "matched_entity_ids")
        fr_recs = fr.select("entity_id", "src", "country", "business_name", "business_address")
        fr_pairs = pairs.join(fr_recs.filter(pl.col("src") == "S1").select(pl.col("entity_id").alias("s1")),
                              on="s1", how="semi")
        print(f"France: {fr_pairs.height:,} predicted pairs; clusters below are pseudo-labels, not ground truth")
        t03.sec_samples(fr_recs, fr_pairs, k=8, k_single=6, seed=11)

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet",
         sys.argv[2] if len(sys.argv) > 2 else "output/matching_results.tsv")
