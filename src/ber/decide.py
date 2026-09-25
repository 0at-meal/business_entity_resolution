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
