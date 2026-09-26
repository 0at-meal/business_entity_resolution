"""T15a: can character n-gram TF-IDF nearest-neighbour search reach the pairs key blocking misses?

Per country, every record becomes "name core + address" text, vectorised as character 3-gram
TF-IDF (hashed, sublinear tf, L2-normalised, so dot product = cosine similarity). Each validation
S1 retrieves its KMAX most similar S2/S3 records within its state (records without a state are
searched from every state), keeping cosine >= THR.

Reports for top-N in TOPNS: recall of never-blocked true pairs, the combined recall ceiling,
and the number of NEW candidates added per S1.

Needs: stage01 (ber-00) and the v3 stage02 candidates (latest ber-01).
Kaggle usage (after bootstrap):
    import t15a_tfidf_probe
    t15a_tfidf_probe.main(DATA)
"""
import gc
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

import t04_val_baselines as t04
from t08_matcher import countries_of, find_cand
from ber.ids import id_code
from ber.metrics import explode_ids
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

TOPNS = [10, 20, 50]
KMAX = 50
THR = 0.3
NFEAT = 2 ** 20
CHUNK = 400_000
COLS = ["id", "src", "country", "name_core", "addr_str", "state"]


def vectorize(texts):
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=NFEAT,
                           alternate_sign=False, norm=None, dtype=np.float32)
    X = sp.vstack([hv.transform(texts[i:i + CHUNK]) for i in range(0, len(texts), CHUNK)]).tocsr()
    return TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(X).astype(np.float32).tocsr()


def search(X, ids, q_idx, c_idx):
    from sparse_dot_topn import sp_matmul_topn
    C = X[c_idx].T.tocsr()
    R = sp_matmul_topn(X[q_idx], C, top_n=KMAX, threshold=THR, sort=True, n_threads=4).tocoo()
    return pl.DataFrame({"s1": ids[q_idx][R.row], "c": ids[c_idx][R.col],
                         "cos": R.data.astype(np.float32)})


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    p_tr = find_checkpoint("train", work)
    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    val = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
             .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect()
             .filter(pl.col("fold") == VAL_FOLD).select("s1", "country"))

    rows, cos_rows = [], []
    for c in countries_of(p_tr):
        t1 = time.time()
        df = (pl.scan_parquet(p_tr).filter(pl.col("country") == c).select(COLS).collect()
                .with_columns(text=pl.col("name_core").list.join(" ") + "  " + pl.col("addr_str").fill_null(""),
                              state=pl.col("state").fill_null("")))
        X = vectorize(df["text"].to_list())
        print(f"  {c}: vectorised {X.shape[0]:,} records ({X.nnz / X.shape[0]:.0f} n-grams each) "
              f"in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
        ids = df["id"].to_numpy()
        src = df["src"].to_numpy()
        state = df["state"].to_numpy()
        is_val = df.select(pl.col("id").is_in(val["s1"].implode())).to_series().to_numpy()
        is_cand = src != 1
        parts = []
        for st in np.unique(state[is_val]):
            q_idx = np.where(is_val & (state == st))[0]
            c_idx = np.where(is_cand & ((state == st) | (state == "")))[0]
            if len(q_idx) and len(c_idx):
                parts.append(search(X, ids, q_idx, c_idx))
        tf = (pl.concat(parts)
                .with_columns(r=pl.col("cos").rank("ordinal", descending=True).over("s1")))
        del X, df, parts
        gc.collect()
        print(f"  {c}: search done, {tf.height:,} pairs, {time.time() - t1:.0f}s {t04.mem()}", flush=True)

        val_c = val.filter(pl.col("country") == c).select("s1")
        cand = pl.read_parquet(find_cand("train", c, work)).join(val_c, on="s1", how="semi").select("s1", "c")
        gv = gt.join(val_c, on="s1", how="semi")
        nb = gv.join(cand, on=["s1", "c"], how="anti")
        found_before = gv.height - nb.height
        for n in TOPNS:
            t = tf.filter(pl.col("r") <= n).select("s1", "c")
            reach = nb.join(t, on=["s1", "c"], how="semi")
            new = t.join(cand, on=["s1", "c"], how="anti")
            rows.append({
                "country": c, "top_n": n,
                "tfidf_alone_recall": gv.join(t, on=["s1", "c"], how="semi").height / gv.height,
                "not_blocked": nb.height,
                "reached": reach.height,
                "reach_share_of_nb": reach.height / max(nb.height, 1),
                "new_per_s1": new.height / val_c.height,
                "new_true_rate": new.join(gv, on=["s1", "c"], how="semi").height / max(new.height, 1),
                "ceiling_now": found_before / gv.height,
                "ceiling_with_tfidf": (found_before + reach.height) / gv.height,
            })
        nbc = nb.join(tf, on=["s1", "c"], how="left")
        cos_rows.append({"country": c,
                         "nb_found_any_rank": nbc["cos"].is_not_null().mean(),
                         "nb_cos_p25": nbc["cos"].quantile(0.25), "nb_cos_p50": nbc["cos"].median(),
                         "nb_rank_p50": nbc["r"].median(), "nb_rank_p90": nbc["r"].quantile(0.9)})
        del tf, cand, nbc
        gc.collect()

    t04.section("1. TF-IDF NEAREST NEIGHBOURS vs KEY BLOCKING (validation)")
    res = pl.DataFrame(rows)
    print(res)
    for n in TOPNS:
        r = res.filter(pl.col("top_n") == n)
        nbt, rt = r["not_blocked"].sum(), r["reached"].sum()
        gvt = (r["not_blocked"] / (1 - r["ceiling_now"])).sum()
        print(f"ALL top-{n}: reached {rt:,} of {nbt:,} never-blocked ({rt / nbt:.3f}); "
              f"ceiling {1 - nbt / gvt:.4f} -> {1 - (nbt - rt) / gvt:.4f}")
    t04.section("2. NEVER-BLOCKED TRUE PAIRS: how similar are they? (cos >= threshold only)")
    print(pl.DataFrame(cos_rows))
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
