"""Compact integer encoding of entity ids: 'S2-403571744' -> 20403571744 (Int64)."""
import polars as pl

SRC_MULT = 10_000_000_000


def id_code(col="entity_id"):
    return (pl.col(col).str.slice(1, 1).cast(pl.Int64) * SRC_MULT
            + pl.col(col).str.extract(r"(\d+)$", 1).cast(pl.Int64))


def src_of(col="id"):
    return (pl.col(col) // SRC_MULT).cast(pl.Int8)
