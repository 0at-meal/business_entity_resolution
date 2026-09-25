"""Multi-pass blocking over stage-01 records (v3). Every pass runs within one country.

Each pass maps records to hashed keys; S1 and S2/S3 records sharing a key become candidates.
Keys shared by more than `cap` S2/S3 records are skipped. The union keeps a bitmask of passes.

v3 changes (from T09 error analysis):
  * rare-token passes ignore tokens that occur only once in the country (typo-made tokens
    can never match, but used to crowd out useful rare tokens)
  * digit-for-letter swaps inside words are undone for keys (g1obal -> global, 8lue -> blue)
  * number keys also use leading-digit-truncated variants (704 <-> 04); the model decides
  * new pass name_prefix: state + first 3 letters of the first two name words
"""
import gc
import time

import polars as pl

from .normalize import num_ext

PASSES = [  # (name, cap) ; bit i = 1 << i
    ("name_exact", 300),
    ("num_name", 50),
    ("num_addr", 50),
    ("name_pair", 50),
    ("name_prefix", 50),
]
RARE_NAME = 2
RARE_ADDR = 2
RARE_PAIR = 3
DIGIT_FIX = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b"}


def _fix_core(col="name_core"):
    el = pl.element()
    fixed = el
    for d, ch in DIGIT_FIX.items():
        fixed = fixed.str.replace_all(d, ch, literal=True)
    return pl.col(col).list.eval(pl.when(el.str.contains("[a-z]")).then(fixed).otherwise(el))


def prepare(df):
    return (df.with_columns(core_f=_fix_core())
              .with_columns(k1f=pl.col("core_f").list.join(""),
                            k2f=pl.col("core_f").list.sort().list.join(" "),
                            numx=num_ext(pl.col("num"))))


def _rarest(df, col, n):
    """(id, tok): each record's n rarest tokens (len >= 3) among tokens seen at least twice."""
    ex = (df.select("id", tok=pl.col(col).list.unique())
            .explode("tok", empty_as_null=True).drop_nulls("tok")
            .filter(pl.col("tok").str.len_chars() >= 3))
    freq = ex.group_by("tok").agg(pl.len().alias("f")).filter(pl.col("f") >= 2)
    return (ex.join(freq, on="tok")
              .sort(["id", "f", "tok"])
              .group_by("id", maintain_order=True).head(n)
              .select("id", "tok"))


def _keys(df, kind):
    if kind == "name_exact":
        k = (df.select("id", key=pl.concat_list(["k1", "k2", "k1f", "k2f"]).list.unique())
               .explode("key", empty_as_null=True)
               .filter(pl.col("key").is_not_null() & (pl.col("key") != "")))
    elif kind in ("num_name", "num_addr"):
        col, n = ("core_f", RARE_NAME) if kind == "num_name" else ("addr_toks", RARE_ADDR)
        toks = _rarest(df, col, n)
        nums = (df.select("id", num="numx").explode("num", empty_as_null=True)
                  .drop_nulls("num").unique())
        k = (toks.join(nums, on="id")
                 .select("id", key=pl.col("num").cast(pl.String) + "|" + pl.col("tok")))
    elif kind == "name_pair":
        rare = _rarest(df, "core_f", RARE_PAIR).sort(["id", "tok"])
        cnt = rare.group_by("id").agg(pl.len().alias("n"))
        single = (rare.join(cnt.filter(pl.col("n") == 1), on="id", how="semi")
                      .select("id", key=pl.col("tok")))
        multi = (rare.join(cnt.filter(pl.col("n") >= 2), on="id", how="semi")
                     .with_columns(pos=pl.int_range(pl.len()).over("id")))
        pairs = (multi.join(multi, on="id", suffix="_r")
                      .filter(pl.col("pos") < pl.col("pos_r"))
                      .select("id", key=pl.col("tok") + "|" + pl.col("tok_r")))
        k = pl.concat([single, pairs])
    elif kind == "name_prefix":
        a = pl.col("core_f").list.get(0, null_on_oob=True).str.slice(0, 3)
        b = pl.col("core_f").list.get(1, null_on_oob=True).str.slice(0, 3)
        k = (df.select("id", a=a, b=b, st=pl.col("state").fill_null(""))
               .filter(pl.col("a").is_not_null() & (pl.col("a").str.len_chars() >= 2))
               .select("id", key=pl.col("st") + "|" + pl.col("a") + "|" + pl.col("b").fill_null("")))
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
    """df: stage-01 rows of ONE country (id, src, k1, k2, name_core, num, addr_toks, state)."""
    df = prepare(df)
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
