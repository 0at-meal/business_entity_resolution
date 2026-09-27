import polars as pl

def load(path, col):
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    ex = (df.select("source1_entity_id", ids=pl.col(col).fill_null("").str.split(","))
            .explode("ids", empty_as_null=True).filter(pl.col("ids").is_not_null() & (pl.col("ids") != "")))
    return df, ex

s1 = pl.read_parquet("data/parquet/test_source1.parquet", columns=["entity_id"])
s23 = pl.concat([pl.read_parquet(f"data/parquet/test_source{i}.parquet", columns=["entity_id"]) for i in (2, 3)])

for name, path, col in [("matching", "output/matching_results.tsv", "matched_entity_ids"),
                        ("candidates", "output/candidate_pairs.tsv", "candidate_entity_ids")]:
    df, ex = load(path, col)
    print(f"--- {name}: columns {df.columns}, rows {df.height:,}, unique S1 {df['source1_entity_id'].n_unique():,}")
    print("  missing S1 rows:", s1.join(df, left_on="entity_id", right_on="source1_entity_id", how="anti").height)
    print("  extra (unknown) S1 rows:", df.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height)
    print("  bad ID prefix:", ex.filter(~pl.col("ids").str.contains(r"^S[23]-\d+$")).height)
    print("  duplicates within a list:", ex.height - ex.unique().height)
    print("  IDs not in test S2/S3:", ex.join(s23, left_on="ids", right_on="entity_id", how="anti").height)
    if name == "matching":
        m_ex = ex
    else:
        print("  matched pairs NOT in candidates:",
              m_ex.join(ex, on=["source1_entity_id", "ids"], how="anti").height)