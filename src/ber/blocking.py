"""Multi-pass blocking over stage-01 records. Every pass runs within one country.

Each pass maps records to hashed keys; S1 and S2/S3 records sharing a key become candidates.
Keys shared by more than `cap` S2/S3 records are skipped (generic, near-useless for precision).
The union keeps a bitmask saying which passes produced each pair.
"""
import gc

import polars as pl

PASSES = [  # (name, cap) ; bit i = 1 << i
    ("name_exact", 300),
    ("num_name", 200),
    ("num_addr", 200),
    ("name_pair", 100),
]


def _keys(df, kind):
    """df: id, k1, k2, name_core, num, addr_toks (one country). Returns (id, key:u64)."""
    if kind == "name_exact":
        k = (df.select("id", key=pl.concat_list([pl.col("k1"), pl.col("k2")]))
               .explode("key", empty_as_null=True)
               .filter(pl.col("key").is_not_null() & (pl.col("key") != "")))
    elif kind in ("num_name", "num_addr"):
        src = "name_core" if kind == "num_name" else "addr_toks"
        k = (df.select("id", "num",
                       tok=pl.col(src).list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
                                      .list.unique())
               .explode("num", empty_as_null=True).drop_nulls("num")
               .explode("tok", empty_as_null=True).drop_nulls("tok")
               .select("id", key=pl.col("num").cast(pl.String) + "|" + pl.col("tok")))
    elif kind == "name_pair":
        t = df.select("id", tok=pl.col("name_core").list.unique().list.sort().list.head(6))
        single = (t.filter(pl.col("tok").list.len() == 1)
                   .select("id", key=pl.col("tok").list.first()))
        multi = (t.filter(pl.col("tok").list.len() >= 2)
                  .explode("tok", empty_as_null=True)
                  .with_columns(pos=pl.int_range(pl.len()).over("id")))
        pairs = (multi.join(multi, on="id", suffix="_r")
                      .filter(pl.col("pos") < pl.col("pos_r"))
                      .select("id", key=pl.col("tok") + "|" + pl.col("tok_r")))
        k = pl.concat([single, pairs])
    else:
        raise ValueError(kind)
    return k.select("id", key=pl.col("key").hash()).unique()


def _pairs(ks1, kc, cap):
    bs = kc.group_by("key").agg(pl.len().alias("bs"))
    big = bs.filter(pl.col("bs") > cap).select("key")
    a = ks1.join(big, on="key", how="anti")
    b = kc.join(big, on="key", how="anti")
    p = (a.rename({"id": "s1"}).join(b.rename({"id": "c"}), on="key")
          .select("s1", "c").unique())
    return p, big.height


def generate(df, passes=PASSES, log=print):
    """df: stage-01 rows of ONE country. Returns (candidates s1,c,mask:u8 ; per-pass stats)."""
    s1 = df.filter(pl.col("src") == 1)
    cd = df.filter(pl.col("src") != 1)
    parts, stats = [], []
    for bit, (kind, cap) in enumerate(passes):
        ks1, kc = _keys(s1, kind), _keys(cd, kind)
        p, n_big = _pairs(ks1, kc, cap)
        parts.append(p.with_columns(m=pl.lit(1 << bit, pl.UInt8)))
        stats.append({"pass": kind, "pairs": p.height, "oversized_keys": n_big})
        del ks1, kc, p
        gc.collect()
    cand = (pl.concat(parts).group_by("s1", "c")
              .agg(pl.col("m").sum().cast(pl.UInt8).alias("mask")))
    return cand, stats
