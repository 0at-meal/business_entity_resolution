"""Leak check (local, read-only): does anything besides the text reveal the matches?

Compares TRUE pairs against RANDOM pairs on ID numbers and file row positions, checks whether
siblings (several matches of one S1 in the same source) sit next to each other in the file, and
whether unmatched records / singletons cluster in a particular ID range or file region.

Usage (from D:\\business_entity_resolution):
    python src/t_leak_check.py data/parquet
"""
import sys

import polars as pl

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/parquet"


def load(src):
    df = pl.read_parquet(f"{DATA}/train_source{src}.parquet", columns=["row_idx", "entity_id", "country"])
    n = df.height
    return df.select(id=pl.col("entity_id"),
                     src=pl.lit(src),
                     num=pl.col("entity_id").str.extract(r"(\d+)$", 1).cast(pl.Int64),
                     row=pl.col("row_idx").cast(pl.Int64),
                     pos=pl.col("row_idx").cast(pl.Float64) / n,
                     country=pl.col("country"))


def pair_stats(P, label):
    return (P.group_by("b_src").agg(
        pl.len().alias("pairs"),
        pl.corr("a_num", "b_num", method="spearman").alias("spearman_id"),
        pl.corr("a_pos", "b_pos", method="spearman").alias("spearman_pos"),
        (pl.col("a_num") - pl.col("b_num")).abs().median().alias("median_abs_id_diff"),
        ((pl.col("a_num") - pl.col("b_num")).abs() < 1_000_000).mean().alias("share_id_within_1e6"),
        (pl.col("a_num") == pl.col("b_num")).mean().alias("share_same_id_num"),
        (pl.col("a_pos") - pl.col("b_pos")).abs().median().alias("median_abs_pos_diff"),
        ((pl.col("a_pos") - pl.col("b_pos")).abs() < 0.001).mean().alias("share_pos_within_0.1pct"),
    ).with_columns(kind=pl.lit(label)).sort("b_src"))


def quantiles(df, col, flag):
    return (df.group_by(flag).agg(
        pl.len().alias("n"),
        *[pl.col(col).quantile(q).alias(f"q{int(q * 100)}") for q in (0.01, 0.1, 0.5, 0.9, 0.99)],
    ).sort(flag))


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")

    s1, s2, s3 = load(1), load(2), load(3)
    cand = pl.concat([s2, s3])
    gt = (pl.read_parquet(f"{DATA}/train_ground_truth.parquet")
            .select(s1=pl.col("source1_entity_id"),
                    c=pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("c", empty_as_null=True)
            .filter(pl.col("c").is_not_null() & (pl.col("c") != "")))
    a = s1.select(pl.col("id").alias("s1"), pl.col("num").alias("a_num"), pl.col("pos").alias("a_pos"))
    b = cand.select(pl.col("id").alias("c"), pl.col("src").alias("b_src"), pl.col("num").alias("b_num"),
                    pl.col("pos").alias("b_pos"), pl.col("row").alias("b_row"))
    P = gt.join(a, on="s1").join(b, on="c")

    rnd = (P.select("s1", "a_num", "a_pos")
             .hstack(b.sample(n=P.height, with_replacement=True, seed=1).select("b_src", "b_num", "b_pos")))

    print("\n=== 1+2. ID NUMBERS AND ROW POSITIONS: true pairs vs random pairs ===")
    print(pl.concat([pair_stats(P, "TRUE"), pair_stats(rnd, "random")]).sort("b_src", "kind"))

    print("\n=== 3. SIBLING ADJACENCY: gaps between rows of one S1's matches in the same source ===")
    sib = (P.sort("s1", "b_src", "b_row")
             .with_columns(gap=pl.col("b_row").diff().over(["s1", "b_src"]))
             .drop_nulls("gap"))
    for src, n in ((2, s2.height), (3, s3.height)):
        g = sib.filter(pl.col("b_src") == src)["gap"]
        print(f"S{src}: {g.len():,} sibling gaps | median {g.median():,.0f} rows "
              f"(random expectation ~{n // 3:,}) | share gap<=1: {(g <= 1).mean():.4f} "
              f"| gap<=10: {(g <= 10).mean():.4f} | gap<=1000: {(g <= 1000).mean():.4f}")

    print("\n=== 4a. MATCHED vs UNMATCHED S2/S3 RECORDS: ID number quantiles ===")
    matched = P.select("c").unique().with_columns(matched=pl.lit(True))
    cf = cand.join(matched, left_on="id", right_on="c", how="left").with_columns(pl.col("matched").fill_null(False))
    print(quantiles(cf, "num", "matched"))
    print("=== 4b. MATCHED vs UNMATCHED: share matched per tenth of the file (row position) ===")
    print(cf.with_columns(decile=(pl.col("pos") * 10).floor().cast(pl.Int8))
            .group_by("src", "decile").agg(pl.col("matched").mean().alias("share_matched"))
            .sort("src", "decile")
            .pivot(on="decile", index="src", values="share_matched"))

    print("\n=== 4c. SINGLETON vs NON-SINGLETON S1: ID number quantiles and share per tenth of file ===")
    has = P.select("s1").unique().with_columns(has_match=pl.lit(True))
    sf = s1.join(has, left_on="id", right_on="s1", how="left").with_columns(pl.col("has_match").fill_null(False))
    print(quantiles(sf, "num", "has_match"))
    print(sf.with_columns(decile=(pl.col("pos") * 10).floor().cast(pl.Int8))
            .group_by("decile").agg((~pl.col("has_match")).mean().alias("share_singleton"))
            .sort("decile").transpose(include_header=True))

    print("\n=== 5. ID NUMBER RANGE per source (min / max / digits) ===")
    print(pl.concat([s1, s2, s3]).group_by("src").agg(
        pl.col("num").min().alias("min"), pl.col("num").max().alias("max"),
        pl.col("id").str.len_chars().min().alias("len_min"), pl.col("id").str.len_chars().max().alias("len_max"))
        .sort("src"))


if __name__ == "__main__":
    main()
