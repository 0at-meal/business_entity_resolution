"""Turn pair probabilities into matches: threshold, optional relative cut, one owner per record."""
import polars as pl

from .metrics import per_entity_f, summarize


def decide(pred, thr, rel=0.0):
    p = pred.filter(pl.col("p") >= thr)
    if rel > 0:
        p = p.filter(pl.col("p") >= rel * pl.col("p").max().over("s1"))
    return (p.sort("p", descending=True)
             .unique(subset=["c"], keep="first", maintain_order=True)
             .select("s1", "c"))


def tune(pred, gt, base, thrs=None, rels=(0.0, 0.3, 0.6)):
    thrs = thrs or [round(x, 2) for x in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    rows = []
    for r in rels:
        for t in thrs:
            e = per_entity_f(decide(pred, t, r), gt, base)
            s = summarize(e).to_dicts()[0]
            rows.append({"thr": t, "rel": r, **s})
    res = pl.DataFrame(rows).sort("F0.5", descending=True)
    best = res.row(0, named=True)
    return best["thr"], best["rel"], res


# --------------------------------------------------------------------------- expected-F (T10)
def decide_expected(pred, alpha=0.0, floor=0.05):
    """Per-S1 set choice maximising expected F0.5.

    1. Each record goes to its highest-probability S1 (one owner per record).
    2. For each S1, candidates are sorted by p. Taking the top k gives expected
       F0.5 ~= 1.25 * sum(p_1..p_k) / (k + 0.25 * E[T]),  E[T] = (1 + alpha) * sum(all p)
       (alpha adds mass for true matches blocking never produced).
    3. Predicting nothing scores 1 only if the S1 is a singleton: P = prod(1 - p_i).
    The option with the higher expected score wins.
    """
    p = (pred.filter(pl.col("p") >= floor)
             .sort("p", descending=True)
             .unique(subset=["c"], keep="first", maintain_order=True)
             .sort(["s1", "p"], descending=[False, True]))
    p = p.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("s1"),
        cum=pl.col("p").cast(pl.Float64).cum_sum().over("s1"),
        et=pl.col("p").cast(pl.Float64).sum().over("s1") * (1.0 + alpha),
        p0=(1.0 - pl.col("p").cast(pl.Float64)).clip(1e-12, 1.0).log().sum().over("s1").exp(),
    ).with_columns(ef=1.25 * pl.col("cum") / (pl.col("k") + 0.25 * pl.col("et")))
    best = (p.group_by("s1")
             .agg(pl.col("k").sort_by("ef", descending=True).first().alias("kbest"),
                  pl.col("ef").max().alias("efmax"), pl.col("p0").first().alias("p0")))
    keep = best.filter(pl.col("efmax") > pl.col("p0")).select("s1", "kbest")
    return (p.join(keep, on="s1").filter(pl.col("k") <= pl.col("kbest")).select("s1", "c"))


def tune_expected(pred, gt, base, alphas=(0.0, 0.05, 0.1, 0.2), floors=(0.02, 0.05, 0.1)):
    rows = []
    for a in alphas:
        for f in floors:
            e = per_entity_f(decide_expected(pred, a, f), gt, base)
            rows.append({"alpha": a, "floor": f, **summarize(e).to_dicts()[0]})
    res = pl.DataFrame(rows).sort("F0.5", descending=True)
    best = res.row(0, named=True)
    return best["alpha"], best["floor"], res
