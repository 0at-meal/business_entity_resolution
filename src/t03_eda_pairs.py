"""T03: EDA on true match pairs (train). Prints a pasteable report.

Kaggle usage (after the bootstrap cell has set DATA and sys.path):
    import importlib, t03_eda_pairs
    importlib.reload(t03_eda_pairs)
    t03_eda_pairs.main(DATA)

Local usage:
    python src/t03_eda_pairs.py data/parquet
"""
import sys
import textwrap
import time

import polars as pl

NONLATIN = ["Devanagari", "Tamil", "Telugu", "Bengali", "Kannada", "Malayalam",
            "Gujarati", "Gurmukhi", "Oriya", "Arabic", "Cyrillic", "Han"]
ACCENTED_LATIN = "[\u00C0-\u024F]"   # letters with diacritics, e.g. é, Ó, ç


# ---------------------------------------------------------------- helpers
def norm(e):
    """Basic cleaning only (for EDA): strip Latin accents, lowercase, keep letters/digits."""
    return (e.str.normalize("NFKD")
             .str.replace_all(r"(\p{Latin})\p{Mn}+", "$1")   # é -> e, but leave Indic vowel signs alone
             .str.to_lowercase()
             .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
             .str.strip_chars())


def jacc(a, b):
    u = a.list.set_union(b).list.len()
    return pl.when(u > 0).then(a.list.set_intersection(b).list.len() / u)


def section(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


# ---------------------------------------------------------------- loading
def load(data):
    frames = []
    for s in ("1", "2", "3"):
        frames.append(
            pl.read_parquet(f"{data}/train_source{s}.parquet",
                            columns=["entity_id", "business_name", "business_address", "country"])
            .with_columns(src=pl.lit("S" + s)))
    recs = pl.concat(frames).with_columns(
        nn=norm(pl.col("business_name")),
        na=norm(pl.col("business_address")),
        nonlatin=pl.col("business_name").str.contains(r"[^\p{Latin}\p{Common}\p{Inherited}]"),
    )
    gt = pl.read_parquet(f"{data}/train_ground_truth.parquet")
    pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
               .explode("matched_entity_ids", empty_as_null=True)
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "c"})
               .filter(pl.col("c").is_not_null() & (pl.col("c") != ""))
               .select("s1", "c"))
    return recs, pairs


# ---------------------------------------------------------------- 1. scripts
def sec_scripts(recs):
    aggs = [pl.len().alias("n"),
            pl.col("business_address").is_null().mean().alias("addr_null"),
            pl.col("business_name").str.contains(ACCENTED_LATIN).mean().alias("nm_accent"),
            pl.col("business_address").str.contains(ACCENTED_LATIN).mean().alias("ad_accent")]
    for sc in NONLATIN:
        aggs.append(pl.col("business_name").str.contains(rf"\p{{{sc}}}").mean().alias(f"nm_{sc}"))
        aggs.append(pl.col("business_address").str.contains(rf"\p{{{sc}}}").mean().alias(f"ad_{sc}"))
    df = recs.group_by("country", "src").agg(aggs).sort("country", "src")
    keep = [c for c in df.columns
            if not (c.startswith(("nm_", "ad_")) and (df[c].max() or 0) < 0.0005)]
    print(df.select(keep))


# ---------------------------------------------------------------- 2. samples
def sec_samples(recs, pairs, k=5, k_single=2, seed=7):
    s1 = recs.filter(pl.col("src") == "S1").select("entity_id", "country")
    sizes = pairs.group_by("s1").agg(pl.len().alias("k"))
    multi = s1.join(sizes, left_on="entity_id", right_on="s1", how="inner").filter(pl.col("k") >= 3)
    single = s1.join(sizes, left_on="entity_id", right_on="s1", how="anti")

    chosen = []
    for country in sorted(s1["country"].unique().to_list()):
        m = multi.filter(pl.col("country") == country)
        s = single.filter(pl.col("country") == country)
        chosen += [(country, i) for i in m.sample(min(k, m.height), seed=seed)["entity_id"].to_list()]
        chosen += [(country, i) for i in s.sample(min(k_single, s.height), seed=seed)["entity_id"].to_list()]

    ids = [i for _, i in chosen]
    members = dict(pairs.filter(pl.col("s1").is_in(ids)).group_by("s1").agg(pl.col("c")).iter_rows())
    all_ids = ids + [c for cs in members.values() for c in cs]
    look = {r["entity_id"]: r for r in
            recs.filter(pl.col("entity_id").is_in(all_ids))
                .select("entity_id", "src", "business_name", "business_address")
                .iter_rows(named=True)}

    def fmt(eid):
        r = look[eid]
        return f"   {r['src']} | {(r['business_name'] or '')[:48]:48s} | {(r['business_address'] or '<null>')[:80]}"

    for country, i in chosen:
        cs = sorted(members.get(i, []))
        print(f"\n[{country}] {i}  ({len(cs)} matches)")
        print(fmt(i))
        for c in cs:
            print(fmt(c))


# ---------------------------------------------------------------- 3. pair similarity
def build_pairs(recs, pairs):
    r = recs.select(
        "entity_id", "src", "country", "nn", "na", "nonlatin",
        nm=pl.col("business_name"),
        nt=pl.col("nn").str.split(" "),
        at=pl.col("na").str.split(" "),
        num=pl.col("business_address").str.extract_all(r"\d+")
              .list.eval(pl.element().str.strip_chars_start("0")).list.unique(),
        has6=pl.col("business_address").str.contains(r"(?:^|\D)\d{6}(?:\D|$)"),
        has5=pl.col("business_address").str.contains(r"(?:^|\D)\d{5}(?:\D|$)"),
    )
    a = r.filter(pl.col("src") == "S1").rename({c: f"a_{c}" for c in r.columns})
    b = r.filter(pl.col("src") != "S1").rename({c: f"b_{c}" for c in r.columns})
    P = (pairs.join(a, left_on="s1", right_on="a_entity_id")
              .join(b, left_on="c", right_on="b_entity_id")
              .with_columns(
                  name_exact=pl.col("a_nn") == pl.col("b_nn"),
                  name_j=jacc(pl.col("a_nt"), pl.col("b_nt")),
                  first_tok=pl.col("a_nt").list.first() == pl.col("b_nt").list.first(),
                  addr_j=jacc(pl.col("a_at"), pl.col("b_at")),
                  num_any=pl.col("a_num").list.set_intersection(pl.col("b_num")).list.len() > 0,
              ))
    return r, P


def sec_pair_stats(P):
    q = lambda c, p: pl.col(c).quantile(p).alias(f"{c}_p{int(p * 100)}")
    stats = (P.group_by("a_country", "b_src", "b_nonlatin")
              .agg(pl.len().alias("n"),
                   pl.col("name_exact").mean(),
                   q("name_j", 0.10), q("name_j", 0.25), q("name_j", 0.50),
                   pl.col("first_tok").mean(),
                   pl.col("b_na").is_null().mean().alias("b_addr_null"),
                   q("addr_j", 0.25), q("addr_j", 0.50),
                   pl.col("num_any").mean(),
                   pl.col("a_has6").mean(), pl.col("b_has6").mean(),
                   pl.col("a_has5").mean(), pl.col("b_has5").mean())
              .sort("a_country", "b_src", "b_nonlatin"))
    print(stats)

    section("3b. NATIVE-SCRIPT CANDIDATES vs THEIR S1 NAME")
    nat = P.filter(pl.col("b_nonlatin"))
    if nat.height:
        print(nat.sample(min(10, nat.height), seed=1)
                 .select(pl.col("a_nm").alias("s1_name"), pl.col("b_src").alias("src"),
                         pl.col("b_nm").alias("candidate_name")))

    section("3c. LOW-NAME-OVERLAP TRUE PAIRS (name_j < 0.3, latin only)")
    low = P.filter((pl.col("name_j") < 0.3) & ~pl.col("b_nonlatin"))
    print(f"share of latin true pairs with name_j < 0.3: {low.height / max(P.filter(~pl.col('b_nonlatin')).height, 1):.4f}")
    if low.height:
        print(low.sample(min(12, low.height), seed=2)
                 .select(pl.col("a_nm").alias("s1_name"), pl.col("b_nm").alias("candidate_name"),
                         pl.col("a_na").str.slice(0, 50).alias("s1_addr"),
                         pl.col("b_na").str.slice(0, 50).alias("cand_addr")))


# ---------------------------------------------------------------- 4. blocking keys
def sec_keys(r, pairs):
    keys = {
        "exact_name": pl.col("nn"),
        "sorted_tokens": pl.col("nt").list.sort().list.join(" "),
        "first_token": pl.col("nt").list.first(),
        "first2_tokens": pl.col("nt").list.head(2).list.join(" "),
    }
    out = []
    for name, expr in keys.items():
        rk = r.select("entity_id", "src", "country", k=expr)
        cc = (rk.filter(pl.col("src") != "S1").group_by("country", "k")
                .agg(pl.len().cast(pl.Int64).alias("n")))
        s1k = (rk.filter(pl.col("src") == "S1").join(cc, on=["country", "k"], how="left")
                 .with_columns(pl.col("n").fill_null(0)))
        gen = s1k.group_by("country").agg(
            pl.len().alias("s1"), pl.col("n").sum().alias("generated"),
            pl.col("n").quantile(0.99).alias("cands_p99"),
            (pl.col("n") == 0).mean().alias("s1_zero_cands"))
        kmap = rk.select("entity_id", "k", "country")
        eq = (pairs.join(kmap, left_on="s1", right_on="entity_id")
                   .join(kmap.select("entity_id", pl.col("k").alias("k_c")), left_on="c", right_on="entity_id")
                   .filter(pl.col("k") == pl.col("k_c"))
                   .group_by("country").agg(pl.len().alias("true_hits")))
        tot = (pairs.join(kmap.select("entity_id", "country"), left_on="s1", right_on="entity_id")
                    .group_by("country").agg(pl.len().alias("true_pairs")))
        mb = cc.group_by("country").agg(pl.col("n").max().alias("max_block"))
        res = (gen.join(eq, on="country", how="left").join(tot, on="country").join(mb, on="country")
                  .with_columns(pl.col("true_hits").fill_null(0))
                  .with_columns(key=pl.lit(name),
                                recall=pl.col("true_hits") / pl.col("true_pairs"),
                                precision=pl.col("true_hits") / pl.col("generated"),
                                cands_per_s1=pl.col("generated") / pl.col("s1")))
        out.append(res.select("key", "country", "recall", "precision", "cands_per_s1",
                              "cands_p99", "s1_zero_cands", "max_block"))
    print(pl.concat(out).sort("country", "key"))


# ---------------------------------------------------------------- 5. tokens
def _top(df, n):
    rows = (df.filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
              .group_by("t").agg(pl.len().alias("c")).sort("c", descending=True).head(n))
    return textwrap.fill("  ".join(f"{t}:{c}" for t, c in rows.iter_rows()), width=170,
                         initial_indent="   ", subsequent_indent="   ")


def sec_tokens(r, per_group=500_000):
    for country in sorted(r["country"].unique().to_list()):
        sub_c = r.filter(pl.col("country") == country)
        for src in ("S1", "S2", "S3"):
            sub = sub_c.filter(pl.col("src") == src)
            sub = sub.sample(min(per_group, sub.height), seed=0)
            print(f"\n[{country} {src}] last name token:")
            print(_top(sub.select(t=pl.col("nt").list.last()), 25))
            print(f"[{country} {src}] last 2 name tokens:")
            print(_top(sub.select(t=pl.col("nt").list.tail(2).list.join(" ")), 15))
        sub = sub_c.sample(min(per_group * 2, sub_c.height), seed=0)
        print(f"\n[{country} ALL] top address tokens (non-numeric):")
        toks = (sub.select(t=pl.col("at")).explode("t", empty_as_null=True)
                   .filter(~pl.col("t").str.contains(r"^\d+$")))
        print(_top(toks, 60))


# ---------------------------------------------------------------- main
def main(data):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(400)
    pl.Config.set_fmt_str_lengths(80)
    pl.Config.set_float_precision(3)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")

    t0 = time.time()
    recs, pairs = load(data)
    print(f"loaded {recs.height:,} records, {pairs.height:,} true pairs in {time.time() - t0:.0f}s")

    section("1. SCRIPTS / ACCENTS / NULLS (share of records)")
    sec_scripts(recs)

    section("2. SAMPLE CLUSTERS")
    sec_samples(recs, pairs)

    r, P = build_pairs(recs, pairs)
    del recs
    section("3. TRUE-PAIR SIMILARITY (basic cleaning only)")
    sec_pair_stats(P)
    del P

    section("4. SIMPLE BLOCKING KEYS (within country)")
    sec_keys(r, pairs)

    section("5. NAME TAIL TOKENS / ADDRESS TOKENS (sampled)")
    sec_tokens(r)

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet")
