"""T15b (A2a): character n-gram TF-IDF nearest-neighbour candidates - blocking pass 6.

One call handles one (split, country[, part]) shard, so the slow search can run in parallel across
notebooks/accounts. Output: stage02t/tf_{split}_{country}_p{part}of{n_parts}.parquet with
columns s1, c, tf_cos, tf_rank (rank 1 = most similar).

Queries:
  train -> the S1 entities the matcher uses: the TRAIN_FRAC training sample + the validation fold
           (same deterministic sample as t08_matcher)
  test  -> every S1 entity
  part/n_parts split the query S1s by id, so one country can be spread over several notebooks.

Kaggle usage (after bootstrap; inputs: ber-parquet, ber-00-bootstrap):
    import t15b_tfidf_block
    t15b_tfidf_block.main(DATA, "test", "India", part=0, n_parts=3)
"""
import gc
import os
import sys
import time

import numpy as np
import polars as pl

import t04_val_baselines as t04
from t08_matcher import TRAIN_FRAC
from t15a_tfidf_probe import COLS, vectorize
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

TOPN = 20
THR = 0.3


def search(X, ids, q_idx, c_idx, topn):
    from sparse_dot_topn import sp_matmul_topn
    C = X[c_idx].T.tocsr()
    R = sp_matmul_topn(X[q_idx], C, top_n=topn, threshold=THR, sort=True, n_threads=4).tocoo()
    return pl.DataFrame({"s1": ids[q_idx][R.row], "c": ids[c_idx][R.col],
                         "tf_cos": R.data.astype(np.float32)})


def query_ids(p, split):
    s1_all = (pl.scan_parquet(p).filter(pl.col("src") == 1)
                .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect())
    if split == "test":
        return s1_all.select("s1")
    val = s1_all.filter(pl.col("fold") == VAL_FOLD)
    trn = s1_all.filter(pl.col("fold") != VAL_FOLD).sample(fraction=TRAIN_FRAC, seed=11)  # same as t08
    return pl.concat([trn, val]).select("s1").unique()


def main(data, split, country, part=0, n_parts=1, work="/kaggle/working", topn=TOPN):
    t0 = time.time()
    os.makedirs(f"{work}/stage02t", exist_ok=True)
    p = find_checkpoint(split, work)
    q = query_ids(p, split).filter((pl.col("s1") // 10 % n_parts) == part)   # id digits are uniform
    df = (pl.scan_parquet(p).filter(pl.col("country") == country).select(COLS).collect()
            .with_columns(text=pl.col("name_core").list.join(" ") + "  " + pl.col("addr_str").fill_null(""),
                          state=pl.col("state").fill_null("")))
    X = vectorize(df["text"].to_list())
    print(f"{split}/{country} part {part}/{n_parts}: vectorised {X.shape[0]:,} records "
          f"in {time.time() - t0:.0f}s {t04.mem()}", flush=True)
    ids = df["id"].to_numpy()
    state = df["state"].to_numpy()
    is_q = df.select(pl.col("id").is_in(q["s1"].implode())).to_series().to_numpy()
    is_c = df["src"].to_numpy() != 1
    del df
    gc.collect()
    states = np.unique(state[is_q])
    print(f"  {int(is_q.sum()):,} query S1s across {len(states)} states", flush=True)
    parts = []
    for i, st in enumerate(states):
        q_idx = np.where(is_q & (state == st))[0]
        c_idx = np.where(is_c & ((state == st) | (state == "")))[0]
        if len(q_idx) and len(c_idx):
            parts.append(search(X, ids, q_idx, c_idx, topn))
        if i % 10 == 0:
            print(f"  state {i + 1}/{len(states)} ({st}) done, {time.time() - t0:.0f}s", flush=True)
    tf = (pl.concat(parts)
            .with_columns(tf_rank=pl.col("tf_cos").rank("ordinal", descending=True).over("s1").cast(pl.UInt8)))
    out = f"{work}/stage02t/tf_{split}_{country}_p{part}of{n_parts}.parquet"
    tf.write_parquet(out, compression="zstd")
    print(f"wrote {out}: {tf.height:,} pairs, {tf.height / max(int(is_q.sum()), 1):.1f} per S1 "
          f"| done in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 0, int(a[5]) if len(a) > 5 else 1)
