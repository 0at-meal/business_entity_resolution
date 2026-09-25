"""Stage 00: convert raw TSVs to zstd Parquet and print a data profile.

Usage:
    python src/stage00_convert_profile.py --raw-dir data/raw --out-dir data/parquet
"""
import argparse
import json
import os
import time

import polars as pl

SOURCES = ["source1", "source2", "source3"]


def count_raw_rows(path):
    """Count newline-terminated lines minus header (handles missing trailing newline)."""
    n, last = 0, b"\n"
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def convert(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log = {}
    for split in ["train", "test"]:
        files = SOURCES + (["ground_truth"] if split == "train" else [])
        for f in files:
            src = os.path.join(raw_dir, split, f"{split}_{f}.tsv")
            dst = os.path.join(out_dir, f"{split}_{f}.parquet")
            t0 = time.time()
            lf = pl.scan_csv(
                src,
                separator="\t",
                quote_char=None,          # never treat " as a quote in addresses
                infer_schema_length=0,    # everything as string
                encoding="utf8-lossy",
            )
            if f != "ground_truth":
                lf = lf.with_row_index("row_idx")
            lf.sink_parquet(dst, compression="zstd")
            rows = pl.scan_parquet(dst).select(pl.len()).collect().item()
            raw = count_raw_rows(src)
            key = f"{split}_{f}"
            log[key] = {
                "rows": rows,
                "raw_lines": raw,
                "tsv_mb": round(os.path.getsize(src) / 1e6, 1),
                "parquet_mb": round(os.path.getsize(dst) / 1e6, 1),
                "secs": round(time.time() - t0, 1),
            }
            flag = "" if rows == raw else "   <-- ROW COUNT MISMATCH"
            print(f"[convert] {key}: {rows:,} rows (raw {raw:,}) "
                  f"{log[key]['tsv_mb']}MB -> {log[key]['parquet_mb']}MB "
                  f"in {log[key]['secs']}s{flag}")
    return log


def _len_stats(s):
    L = s.str.len_chars().drop_nulls()
    if L.len() == 0:
        return None
    return {"mean": round(L.mean(), 1), "p50": L.median(),
            "p99": L.quantile(0.99), "max": L.max()}


def profile_source(df):
    out = {
        "rows": df.height,
        "columns": df.columns,
        "nulls": {c: int(df[c].null_count()) for c in df.columns},
        "dup_entity_ids": int(df["entity_id"].is_duplicated().sum()),
        "id_prefixes": dict(df["entity_id"].str.slice(0, 3)
                            .value_counts(sort=True).iter_rows()),
        "country_top10": dict(df["country"].value_counts(sort=True)
                              .head(10).iter_rows()),
        "name_len": _len_stats(df["business_name"]),
        "address_len": _len_stats(df["business_address"]),
        "sample": df.sample(3, seed=0).drop("row_idx").to_dicts(),
    }
    return out


def profile_ground_truth(pq):
    s1 = pl.read_parquet(f"{pq}/train_source1.parquet", columns=["entity_id", "country"])
    s2 = pl.read_parquet(f"{pq}/train_source2.parquet", columns=["entity_id", "country"])
    s3 = pl.read_parquet(f"{pq}/train_source3.parquet", columns=["entity_id", "country"])
    gt = pl.read_parquet(f"{pq}/train_ground_truth.parquet")

    gt = gt.with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",")
        .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        .alias("matches")
    )
    sizes = gt["matches"].list.len()
    pairs = (gt.select("source1_entity_id", "matches")
               .explode("matches", empty_as_null=True).drop_nulls("matches")
               .rename({"matches": "cand_id"}))
    cand = pl.concat([s2, s3]).rename({"entity_id": "cand_id", "country": "cand_country"})
    joined = (pairs.join(s1.rename({"entity_id": "source1_entity_id",
                                    "country": "s1_country"}),
                         on="source1_entity_id", how="left")
                   .join(cand, on="cand_id", how="left"))
    both = joined.drop_nulls(["s1_country", "cand_country"])

    matched_s2 = pairs.filter(pl.col("cand_id").str.starts_with("S2-"))["cand_id"].n_unique()
    matched_s3 = pairs.filter(pl.col("cand_id").str.starts_with("S3-"))["cand_id"].n_unique()

    return {
        "gt_rows": gt.height,
        "train_s1_rows": s1.height,
        "s1_missing_from_gt": int((~s1["entity_id"].is_in(gt["source1_entity_id"].implode())).sum()),
        "dup_s1_rows_in_gt": int(gt["source1_entity_id"].is_duplicated().sum()),
        "singleton_frac": round(float((sizes == 0).mean()), 4),
        "match_list_size_dist": dict(
            sizes.clip(upper_bound=10).value_counts(sort=True).sort(sizes.name).iter_rows()),
        "total_pairs": pairs.height,
        "pairs_to_S2": int(pairs["cand_id"].str.starts_with("S2-").sum()),
        "pairs_to_S3": int(pairs["cand_id"].str.starts_with("S3-").sum()),
        "cand_ids_claimed_by_multiple_s1": int(pairs["cand_id"].is_duplicated().sum()),
        "gt_ids_not_found_in_sources": int(joined["cand_country"].is_null().sum()),
        "frac_S2_records_matched": round(matched_s2 / max(s2.height, 1), 4),
        "frac_S3_records_matched": round(matched_s3 / max(s3.height, 1), 4),
        "pair_country_agreement": round(
            float((both["s1_country"] == both["cand_country"]).mean()), 4)
            if both.height else None,
        "sample_rows": gt.filter(sizes > 0).sample(3, seed=0)
                         .select("source1_entity_id", "matched_entity_ids").to_dicts(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/parquet")
    ap.add_argument("--report", default="reports/profile_v1.json")
    args = ap.parse_args()

    report = {"conversion": convert(args.raw_dir, args.out_dir), "sources": {}}
    for split in ["train", "test"]:
        for f in SOURCES:
            key = f"{split}_{f}"
            df = pl.read_parquet(os.path.join(args.out_dir, f"{key}.parquet"))
            report["sources"][key] = profile_source(df)
            del df
    report["ground_truth"] = profile_ground_truth(args.out_dir)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)
    print("\n===== PROFILE (paste everything below back) =====")
    print(json.dumps({k: v for k, v in report.items() if k != "conversion"},
                     indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
