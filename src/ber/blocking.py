"""Multi-pass blocking over stage-01 records. Every pass runs within one country.

Each pass maps records to hashed keys; S1 and S2/S3 records sharing a key become candidates.
To keep volume bounded, token-based passes only use each record's RAREST tokens (document
frequency counted within the country), and keys shared by more than `cap` S2/S3 records are skipped.
The union keeps a bitmask saying which passes produced each pair.
"""
import gc
import time

import polars as pl

PASSES = [  # (name, cap) ; bit i = 1 << i
    ("name_exact", 300),
    ("num_name", 50),
    ("num_addr", 50),
    ("name_pair", 50),
]
RARE_NAME = 2   # rarest name tokens per record used by num_name
RARE_ADDR = 2   # rarest address tokens per record used by num_addr
RARE_PAIR = 3   # rarest name tokens per record combined pairwise by name_pair


def _rarest(df, col, n):
    """(id, tok) for each record's n rarest tokens of length >= 3 (ties broken alphabetically)."""
    ex = (df.select("id", tok=pl.col(col).list.unique())
            .explode("tok", empty_as_null=True).drop_nulls("tok")
            .filter(pl.col("tok").str.len_chars() >= 3))
    freq = ex.group_by("tok").agg(pl.len().alias("f"))
    return (ex.join(freq, on="tok")
              .sort(["id", "f", "tok"])
              .group_by("id", maintain_order=True).head(n)
              .select("id", "tok"))


def _keys(df, kind):
    """df: id, k1, k2, name_core, num, addr_toks (one country, all sources). Returns (id, key:u64)."""
    if kind == "name_exact":
        k = (df.select("id", key=pl.concat_list([pl.col("k1"), pl.col("k2")]))
               .explode("key", empty_as_null=True)
               .filter(pl.col("key").is_not_null() & (pl.col("key") != "")))
    elif kind in ("num_name", "num_addr"):
        col, n = ("name_core", RARE_NAME) if kind == "num_name" else ("addr_toks", RARE_ADDR)
        toks = _rarest(df, col, n)
        nums = (df.select("id", "num").explode("num", empty_as_null=True).drop_nulls("num")
                  .unique())
        k = (toks.join(nums, on="id")
                 .select("id", key=pl.col("num").cast(pl.String) + "|" + pl.col("tok")))
    elif kind == "name_pair":
        rare = _rarest(df, "name_core", RARE_PAIR).sort(["id", "tok"])
        cnt = rare.group_by("id").agg(pl.len().alias("n"))
        single = (rare.join(cnt.filter(pl.col("n") == 1), on="id", how="semi")
                      .select("id", key=pl.col("tok")))
        multi = (rare.join(cnt.filter(pl.col("n") >= 2), on="id", how="semi")
                     .with_columns(pos=pl.int_range(pl.len()).over("id")))
        pairs = (multi.join(multi, on="id", suffix="_r")
                      .filter(pl.col("pos") < pl.col("pos_r"))
                      .select("id", key=pl.col("tok") + "|" + pl.col("tok_r")))
        k = pl.concat([single, pairs])
    else:
        raise ValueError(kind)
    return k.select("id", key=pl.col("key").hash()).unique()


def _pairs(ks1, kc, cap):
    big = (kc.group_by("key").agg(pl.len().alias("bs"))
             .filter(pl.col("bs") > cap).select("key"))
    a = ks1.join(big, on="key", how="anti")
    b = kc.join(big, on="key", how="anti")
    p = (a.rename({"id": "s1"}).join(b.rename({"id": "c"}), on="key")
          .select("s1", "c").unique())
    return p, big.height


def _rss():
    try:
        import psutil
        return f"[RSS {psutil.Process().memory_info().rss / 1e9:.1f} GB]"
    except Exception:
        return ""


def generate(df, passes=PASSES, log=print):
    """df: stage-01 rows of ONE country. Returns (candidates s1,c,mask:u8 ; per-pass stats)."""
    src = df.select("id", "src")
    parts, stats = [], []
    for bit, (kind, cap) in enumerate(passes):
        t1 = time.time()
        keys = _keys(df, kind).join(src, on="id")
        ks1 = keys.filter(pl.col("src") == 1).select("id", "key")
        kc = keys.filter(pl.col("src") != 1).select("id", "key")
        del keys
        p, n_big = _pairs(ks1, kc, cap)
        parts.append(p.with_columns(m=pl.lit(1 << bit, pl.UInt8)))
        stats.append({"pass": kind, "pairs": p.height, "oversized_keys": n_big})
        log(f"    pass {kind}: {p.height:,} pairs ({n_big:,} oversized keys) "
            f"in {time.time() - t1:.0f}s {_rss()}", flush=True)
        del ks1, kc, p
        gc.collect()
    cand = (pl.concat(parts).group_by("s1", "c")
              .agg(pl.col("m").sum().cast(pl.UInt8).alias("mask")))
    return cand, stats
