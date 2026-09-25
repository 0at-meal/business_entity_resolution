"""Deterministic train/validation folds from the numeric part of the S1 entity_id."""
import polars as pl

N_FOLDS = 10
VAL_FOLD = 0


def fold_expr(col="entity_id", k=N_FOLDS):
    return (pl.col(col).str.extract(r"(\d+)$", 1).cast(pl.Int64) % k).alias("fold")
