"""Stage 01: normalized record table (names + addresses), saved as a reusable checkpoint.

Columns:
    id (Int64), entity_id, src (1/2/3), country, nonlatin,
    name_toks (cleaned tokens, native-script mapped), name_core (legal/filler removed),
    k1 (core joined, no spaces), k2 (core sorted, space-joined), name_str,
    num (UInt32 house numbers), addr_null, state, addr_toks, addr_str
"""
import gc
import glob
import os
import time

import polars as pl

from .address import add_address
from .ids import id_code
from .normalize import (addr_numbers_int, apply_native_dict, core_from_tokens_v1, is_nonlatin,
                        name_tokens_v1)


def build_base(data, split, after_markers=None, before_markers=None, keep_raw=False, log=print):
    """Everything except native-script mapping and name keys."""
    frames = []
    for s in ("1", "2", "3"):
        t1 = time.time()
        raw = pl.read_parquet(f"{data}/{split}_source{s}.parquet",
                              columns=["entity_id", "business_name", "business_address", "country"])
        df = raw.select(
            id=id_code(),
            entity_id=pl.col("entity_id"),
            src=pl.lit(int(s), pl.Int8),
            country=pl.col("country"),
            name_toks=name_tokens_v1(pl.col("business_name"), after_markers, before_markers),
            nonlatin=is_nonlatin(pl.col("business_name")).fill_null(False),
            num=addr_numbers_int(pl.col("business_address")),
            addr_null=pl.col("business_address").is_null(),
            business_address=pl.col("business_address"),
            business_name=pl.col("business_name"),
        )
        del raw
        parts = [add_address(df.filter(pl.col("country") == c), c)
                 for c in sorted(df["country"].unique().to_list())]
        df = pl.concat(parts)
        if not keep_raw:
            df = df.drop("business_address", "business_name")
        frames.append(df)
        del parts, df
        gc.collect()
        log(f"  base {split}_source{s} in {time.time() - t1:.0f}s")
    return pl.concat(frames)


def finalize(base, native_dict):
    """Map native-script tokens, then derive name core and keys."""
    tok, stats = apply_native_dict(base.select("id", pl.col("name_toks").alias("toks"), "nonlatin"),
                                   native_dict)
    out = (base.drop("name_toks")
               .join(tok.select("id", pl.col("toks_nat").alias("name_toks")), on="id", how="left")
               .with_columns(name_core=core_from_tokens_v1(pl.col("name_toks")))
               .with_columns(k1=pl.col("name_core").list.join(""),
                             k2=pl.col("name_core").list.sort().list.join(" "),
                             name_str=pl.col("name_toks").list.join(" ")))
    return out, stats


def find_checkpoint(split, work="/kaggle/working"):
    """Locate a saved stage01 file: this session's working dir first, then any attached input."""
    candidates = [f"{work}/stage01/{split}.parquet"] + \
        glob.glob(f"/kaggle/input/**/stage01/{split}.parquet", recursive=True)
    for p in candidates:
        if os.path.exists(p):
            return p
    return None
