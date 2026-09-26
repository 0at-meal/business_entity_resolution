"""Pair features for the matcher.

Phase A (cheap): 3 rapidfuzz similarities -> quick score q -> keep top-K candidates per S1.
Context: ranks/gaps of q within each S1 and within each candidate (competition between S1s).
Phase B (full): name/address/number/state features on the top-K pairs.
Country is intentionally NOT a feature (test contains an unseen country).
"""
import gc

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .normalize import num_ext

REC_COLS = ["id", "src", "nonlatin", "k1", "name_str", "name_core", "addr_toks", "addr_str",
            "num", "state", "addr_null"]
CHEAP = ["n_ratio", "n_tsr", "a_tsr", "q"]
CTX = ["n_cand_s1", "q_rank_s1", "q_gap_s1", "n_cand_c", "q_rank_c", "q_gap_c"]
FULL = ["k1_eq", "k1_empty", "n_jw", "n_tsort", "n_partial", "core_j", "core_cont", "a_core_n",
        "b_core_n", "addr_j", "addr_cont", "a_addr_n", "b_addr_n", "a_ratio", "num_inter",
        "num_trunc", "a_num_n", "b_num_n", "state_rel", "b_addr_null", "b_nonlatin", "b_src",
        "m0", "m1", "m2", "m3", "m4",
        # v2 (T10): residual words = words one side has that the other lacks
        "n_resid_ratio", "a_resid_n", "b_resid_n", "addr_resid_ratio", "num_conflict",
        # v3 (T16): TF-IDF nearest-neighbour blocking pass (bit 32)
        "m5", "tf_cos", "tf_rank"]
TF_COLS = ["tf_cos", "tf_rank"]
KEEP_BIT = 32   # pairs found by the TF-IDF pass always survive the top-K cut
FEATS = CHEAP + CTX + FULL


def _jacc(a, b):
    u = a.list.set_union(b).list.len()
    return pl.when(u > 0).then(a.list.set_intersection(b).list.len() / u).otherwise(0.0)


def _cont(a, b):
    m = pl.min_horizontal(a.list.len(), b.list.len())
    return pl.when(m > 0).then(a.list.set_intersection(b).list.len() / m).otherwise(0.0)


def _rf(df, ca, cb, scorer):
    return pl.Series(cpdist(df[ca].fill_null("").to_numpy(), df[cb].fill_null("").to_numpy(),
                            scorer=scorer, workers=-1, dtype=np.float32))


def attach(pairs, recs):
    a = recs.rename({c: f"a_{c}" for c in recs.columns})
    b = recs.rename({c: f"b_{c}" for c in recs.columns})
    return pairs.join(a, left_on="s1", right_on="a_id").join(b, left_on="c", right_on="b_id")


def cheap(p):
    """Quick score q in [0, 2]. When either address is missing, the name score counts twice
    (otherwise address-less records sink below the top-K cut - see T09)."""
    p = p.with_columns(_anull=pl.col("a_addr_str").fill_null("").eq("")
                       | pl.col("b_addr_str").fill_null("").eq(""))
    p = p.with_columns(n_ratio=_rf(p, "a_k1", "b_k1", fuzz.ratio),
                       n_tsr=_rf(p, "a_name_str", "b_name_str", fuzz.token_set_ratio),
                       a_tsr=_rf(p, "a_addr_str", "b_addr_str", fuzz.token_set_ratio))
    nbest = pl.max_horizontal("n_ratio", "n_tsr")
    return p.with_columns(
        q=((nbest + pl.when(pl.col("_anull")).then(nbest).otherwise(pl.col("a_tsr"))) / 100.0)
        .cast(pl.Float32))


def phase_a(cand, recs, k, chunk=40_000):
    """cand: s1, c, mask (+ tf_cos, tf_rank). Returns top-k pairs per S1 by quick score, plus every
    pair found by the TF-IDF pass (found precisely because keys missed them, so q may be low)."""
    rc = recs.select("id", "k1", "name_str", "addr_str")
    extra = [c for c in TF_COLS if c in cand.columns]
    s1s = cand.select("s1").unique().sort("s1")
    outs = []
    for i in range(0, s1s.height, chunk):
        ids = s1s.slice(i, chunk)
        p = cheap(attach(cand.join(ids, on="s1", how="semi"), rc))
        p = p.filter((pl.col("q").rank("ordinal", descending=True).over("s1") <= k)
                     | ((pl.col("mask") & KEEP_BIT) > 0))
        outs.append(p.select("s1", "c", "mask", *extra, *CHEAP))
        del p
        gc.collect()
    return pl.concat(outs)


def context(p):
    q = pl.col("q")
    return p.with_columns(
        n_cand_s1=pl.len().over("s1").cast(pl.Float32),
        q_rank_s1=q.rank("ordinal", descending=True).over("s1").cast(pl.Float32),
        q_gap_s1=(q - q.max().over("s1")).cast(pl.Float32),
        n_cand_c=pl.len().over("c").cast(pl.Float32),
        q_rank_c=q.rank("ordinal", descending=True).over("c").cast(pl.Float32),
        q_gap_c=(q - q.max().over("c")).cast(pl.Float32),
    )


def full(p, recs):
    for col in TF_COLS:
        if col not in p.columns:
            p = p.with_columns(pl.lit(None, dtype=pl.Float32).alias(col))
    p = attach(p, recs.select(REC_COLS))
    p = p.with_columns(
        k1_eq=(pl.col("a_k1") == pl.col("b_k1")),
        k1_empty=(pl.col("a_k1") == "") | (pl.col("b_k1") == ""),
        core_j=_jacc(pl.col("a_name_core"), pl.col("b_name_core")),
        core_cont=_cont(pl.col("a_name_core"), pl.col("b_name_core")),
        a_core_n=pl.col("a_name_core").list.len(), b_core_n=pl.col("b_name_core").list.len(),
        addr_j=_jacc(pl.col("a_addr_toks"), pl.col("b_addr_toks")),
        addr_cont=_cont(pl.col("a_addr_toks"), pl.col("b_addr_toks")),
        a_addr_n=pl.col("a_addr_toks").list.len(), b_addr_n=pl.col("b_addr_toks").list.len(),
        num_inter=pl.col("a_num").list.set_intersection(pl.col("b_num")).list.len(),
        num_trunc=(num_ext(pl.col("a_num")).list.set_intersection(num_ext(pl.col("b_num")))
                   .list.len() > 0),
        a_num_n=pl.col("a_num").list.len(), b_num_n=pl.col("b_num").list.len(),
        state_rel=pl.when(pl.col("a_state").is_null() | pl.col("b_state").is_null()).then(-1)
                    .when(pl.col("a_state") == pl.col("b_state")).then(1).otherwise(0),
        b_addr_null=pl.col("b_addr_null"), b_nonlatin=pl.col("b_nonlatin"), b_src=pl.col("b_src"),
        m0=(pl.col("mask") & 1) > 0, m1=(pl.col("mask") & 2) > 0,
        m2=(pl.col("mask") & 4) > 0, m3=(pl.col("mask") & 8) > 0, m4=(pl.col("mask") & 16) > 0, m5=(pl.col("mask") & 32) > 0,
        _nra=pl.col("a_name_core").list.set_difference(pl.col("b_name_core")).list.join(" "),
        _nrb=pl.col("b_name_core").list.set_difference(pl.col("a_name_core")).list.join(" "),
        _ara=pl.col("a_addr_toks").list.set_difference(pl.col("b_addr_toks")).list.join(" "),
        _arb=pl.col("b_addr_toks").list.set_difference(pl.col("a_addr_toks")).list.join(" "),
        a_resid_n=pl.col("a_name_core").list.set_difference(pl.col("b_name_core")).list.len(),
        b_resid_n=pl.col("b_name_core").list.set_difference(pl.col("a_name_core")).list.len(),
    )
    p = p.with_columns(num_conflict=(pl.col("a_num_n") > 0) & (pl.col("b_num_n") > 0)
                       & (pl.col("num_inter") == 0) & ~pl.col("num_trunc"))
    p = p.with_columns(n_jw=_rf(p, "a_k1", "b_k1", JaroWinkler.normalized_similarity),
                       n_tsort=_rf(p, "a_name_str", "b_name_str", fuzz.token_sort_ratio),
                       n_partial=_rf(p, "a_k1", "b_k1", fuzz.partial_ratio),
                       a_ratio=_rf(p, "a_addr_str", "b_addr_str", fuzz.ratio),
                       n_resid_ratio=_rf(p, "_nra", "_nrb", fuzz.ratio),
                       addr_resid_ratio=_rf(p, "_ara", "_arb", fuzz.ratio))
    return p.select("s1", "c", *[pl.col(f).fill_null(0).cast(pl.Float32) for f in FEATS])


def full_chunked(p, recs, chunk=100_000):
    s1s = p.select("s1").unique().sort("s1")
    outs = []
    for i in range(0, s1s.height, chunk):
        ids = s1s.slice(i, chunk)
        outs.append(full(p.join(ids, on="s1", how="semi"), recs))
        gc.collect()
    return pl.concat(outs) if outs else None
