"""Macro-averaged F-beta exactly as the challenge defines it (per S1 entity, singletons included)."""
import polars as pl

EMPTY_PAIRS = pl.DataFrame(schema={"s1": pl.String, "c": pl.String})


def explode_ids(df, id_col, list_col):
    """(id, 'a,b,c') rows -> (s1, c) pairs; empty/null lists produce no rows."""
    return (df.select(pl.col(id_col).alias("s1"),
                      pl.col(list_col).fill_null("").str.split(",").alias("c"))
              .explode("c", empty_as_null=True)
              .with_columns(pl.col("c").str.strip_chars())
              .filter(pl.col("c").is_not_null() & (pl.col("c") != "")))


def per_entity_f(pred, true, base, beta=0.5):
    """pred/true: frames with columns s1, c. base: frame with column s1 (+ any extra columns
    such as country) listing every evaluated S1 entity. Returns base with tp/n_pred/n_true/f/prec/rec."""
    b2 = beta * beta
    ids = base.select("s1").unique()
    p = pred.select("s1", "c").join(ids, on="s1", how="semi").unique()
    t = true.select("s1", "c").join(ids, on="s1", how="semi").unique()
    tp = p.join(t, on=["s1", "c"], how="semi").group_by("s1").agg(pl.len().alias("tp"))
    n_pred = p.group_by("s1").agg(pl.len().alias("n_pred"))
    n_true = t.group_by("s1").agg(pl.len().alias("n_true"))
    e = (base.join(tp, on="s1", how="left")
             .join(n_pred, on="s1", how="left")
             .join(n_true, on="s1", how="left")
             .with_columns(pl.col("tp", "n_pred", "n_true").fill_null(0).cast(pl.Float64)))
    P = pl.col("tp") / pl.col("n_pred")
    R = pl.col("tp") / pl.col("n_true")
    f = (pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise((1 + b2) * P * R / (b2 * P + R)))
    return e.with_columns(f=f,
                          prec=pl.when(pl.col("n_pred") > 0).then(P),
                          rec=pl.when(pl.col("n_true") > 0).then(R))


def summarize(e, by=None):
    aggs = [pl.len().alias("n"),
            pl.col("f").mean().alias("F0.5"),
            pl.col("prec").mean().alias("P_avg"),
            pl.col("rec").mean().alias("R_avg"),
            (pl.col("n_pred") > 0).mean().alias("pred_nonempty"),
            (pl.col("n_true") == 0).mean().alias("singleton_frac"),
            pl.col("f").filter(pl.col("n_true") == 0).mean().alias("F_singletons"),
            pl.col("f").filter(pl.col("n_true") > 0).mean().alias("F_nonsingle")]
    if by:
        return e.group_by(by).agg(aggs).sort(by)
    return e.select(aggs)


def self_test():
    """Checks against the worked example in the problem statement."""
    base = pl.DataFrame({"s1": ["S1-1", "S1-2", "S1-3", "S1-4"]})
    true = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-2"], "c": ["S2-47", "S3-812", "S3-4"]})
    pred = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-1", "S1-4"],
                         "c": ["S2-47", "S2-193", "S3-812", "S2-9"]})
    e = per_entity_f(pred, true, base).sort("s1")
    got = e["f"].to_list()
    # S1-1: example -> 0.714 ; S1-2: missed -> 0 ; S1-3: correct singleton -> 1 ; S1-4: false merge on singleton -> 0
    expected = [0.7142857, 0.0, 1.0, 0.0]
    ok = all(abs(a - b) < 1e-6 for a, b in zip(got, expected))
    perfect = per_entity_f(true, true, base)["f"].mean()
    return ok and abs(perfect - 1.0) < 1e-12, got
