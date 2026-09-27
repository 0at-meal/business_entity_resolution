"""T19 (Track C): cross-encoder on EVERY top-10 candidate, then a second-stage model (stacking).

Two steps, so the slow GPU scoring can be sharded across notebooks/accounts:

  1. score(DATA, split, part, n_parts)  [GPU]
       split "val":  top-10 pairs (by T16 LightGBM probability) of every validation S1
       split "test": top-10 pairs of every test S1 (artifacts/test_top10_t16.parquet)
       -> stage03/ce_{split}_p{part}of{n_parts}.parquet  (s1, c, p, ce_<model>)
       French pairs are scored with the French-adapted model when it is attached.

  2. stack(DATA)  [CPU, ~10 min]
       Features per pair: LightGBM logit, cross-encoder logit(s), and each one's rank / gap to the
       best within the same S1. A small LightGBM is trained on validation half A, compared on
       half B against LightGBM alone, then applied to the test set -> output_t19/.

Needs: stage01 (ber-00), artifacts/val_pred_t16 + test_top10_t16 + output_t16 (ber-08),
       ce_model_minilm12v2 (ber-09), optionally ce_model_minilm12v2_fr (Run B).
"""
import gc
import glob
import os
import shutil
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import polars as pl

import t04_val_baselines as t04
from t11b_ce_blend import find_path, logit
from ber.ce import TEXT_COLS, attach_text, predict
from ber.decide import decide, tune
from ber.ids import id_code
from ber.metrics import explode_ids, per_entity_f, summarize
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

SRC_TAG = "t16"
TAG = "t19"
TOPN = 10
CE_NAMES = ["minilm12v2"]
FR_OVERRIDE = {"minilm12v2": "minilm12v2_fr"}


# ----------------------------------------------------------------------------- step 1: scoring
def top_pairs(split, work):
    if split == "val":
        pv = pl.read_parquet(find_path(f"**/artifacts/val_pred_{SRC_TAG}.parquet", work))
        return (pv.filter(pl.col("p").rank("ordinal", descending=True).over("s1") <= TOPN)
                  .select("s1", "c", "p"))
    return pl.read_parquet(find_path(f"**/artifacts/test_top10_{SRC_TAG}.parquet", work)).select("s1", "c", "p")


def load_model(name, work):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = os.path.dirname(find_path(f"**/ce_model_{name}/config.json", work))
    m = AutoModelForSequenceClassification.from_pretrained(d).to("cuda").half()
    if torch.cuda.device_count() > 1:
        m = torch.nn.DataParallel(m)
    print(f"  loaded '{name}' from {d}")
    return m, AutoTokenizer.from_pretrained(d)


def score(data, split, part=0, n_parts=1, work="/kaggle/working", ce_names=None, fr_override=None):
    import torch
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU"
    ce_names = ce_names or CE_NAMES
    fr_override = FR_OVERRIDE if fr_override is None else fr_override
    t0 = time.time()
    os.makedirs(f"{work}/stage03", exist_ok=True)
    top = top_pairs(split, work).filter((pl.col("s1") // 10 % n_parts) == part)
    stage = find_checkpoint("train" if split == "val" else "test", work)
    s1c = (pl.scan_parquet(stage).filter(pl.col("src") == 1)
             .select(pl.col("id").alias("s1"), "country").collect())
    top = top.join(s1c, on="s1")
    ids = pl.concat([top.select(pl.col("s1").alias("id")), top.select(pl.col("c").alias("id"))]).unique()
    text = pl.scan_parquet(stage).select(TEXT_COLS).collect().join(ids, on="id", how="semi")
    df = attach_text(top, text)
    print(f"{split} part {part}/{n_parts}: {df.height:,} pairs ({df['s1'].n_unique():,} S1) {t04.mem()}", flush=True)

    out = df.select("s1", "c", "p", "country")
    for name in ce_names:
        model, tok = load_model(name, work)
        fr_name = fr_override.get(name)
        fr_model = None
        if fr_name and (df["country"] == "France").any():
            try:
                fr_model = load_model(fr_name, work)
            except FileNotFoundError:
                print(f"  French model '{fr_name}' not found - France uses '{name}'")
        col = np.zeros(df.height, dtype=np.float32)
        is_fr = (df["country"] == "France").to_numpy()
        for mask, (m, t) in ((~is_fr, (model, tok)), (is_fr, fr_model or (model, tok))):
            if mask.any():
                idx = np.where(mask)[0]
                t1 = time.time()
                col[idx] = predict(m, t, df["a_text"].gather(idx).to_list(),
                                   df["b_text"].gather(idx).to_list(), 96, bs=2048)
                print(f"  {name}{' (French)' if m is not model else ''}: {len(idx):,} pairs "
                      f"in {time.time() - t1:.0f}s", flush=True)
        out = out.with_columns(pl.Series(f"ce_{name}", col))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    path = f"{work}/stage03/ce_{split}_p{part}of{n_parts}.parquet"
    out.drop("country").write_parquet(path)
    print(f"wrote {path} in {time.time() - t0:.0f}s")


def rescore_france(data, model="minilm12v2_fr2", slot="minilm12v2", work="/kaggle/working"):
    """Rescore every French test top-10 pair with a (newer) French-adapted cross-encoder.
    Writes stage03/fr_override_test.parquet; stack() substitutes it into column ce_<slot>."""
    import torch
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU"
    t0 = time.time()
    os.makedirs(f"{work}/stage03", exist_ok=True)
    stage = find_checkpoint("test", work)
    fr = (pl.scan_parquet(stage).filter((pl.col("src") == 1) & (pl.col("country") == "France"))
            .select(pl.col("id").alias("s1")).collect())
    top = top_pairs("test", work).join(fr, on="s1", how="semi")
    ids = pl.concat([top.select(pl.col("s1").alias("id")), top.select(pl.col("c").alias("id"))]).unique()
    text = pl.scan_parquet(stage).select(TEXT_COLS).collect().join(ids, on="id", how="semi")
    df = attach_text(top, text)
    m, tok = load_model(model, work)
    sc = predict(m, tok, df["a_text"].to_list(), df["b_text"].to_list(), 96, bs=2048)
    out = df.select("s1", "c").with_columns(pl.Series(f"ce_{slot}", sc.astype(np.float32)))
    path = f"{work}/stage03/fr_override_test.parquet"
    out.write_parquet(path)
    print(f"wrote {path}: {out.height:,} French pairs scored with '{model}' in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- step 2: stacking
def load_scores(split, work):
    hits = (glob.glob(f"{work}/stage03/ce_{split}_p*.parquet")
            + glob.glob(f"/kaggle/input/**/stage03/ce_{split}_p*.parquet", recursive=True))
    by_name = {}
    for h in hits:
        by_name.setdefault(os.path.basename(h), h)
    assert by_name, f"no stage03 scores for {split}: attach the scoring notebooks"
    print(f"{split}: {sorted(by_name)}")
    return pl.concat([pl.read_parquet(h) for h in sorted(by_name.values())]).unique(subset=["s1", "c"])


def features(df, names):
    lp = pl.Series("lp", logit(df["p"].to_numpy()).astype(np.float32))
    df = df.with_columns(lp)
    cols = ["lp"]
    exprs = [pl.col("lp").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("lp_rank"),
             (pl.col("lp") - pl.col("lp").max().over("s1")).alias("lp_gap"),
             pl.len().over("s1").cast(pl.Float32).alias("n_top")]
    cols += ["lp_rank", "lp_gap", "n_top"]
    for n in names:
        c = f"ce_{n}"
        exprs += [pl.col(c).rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias(f"{c}_rank"),
                  (pl.col(c) - pl.col(c).max().over("s1")).alias(f"{c}_gap"),
                  (pl.col(c) - pl.col(c).sort(descending=True).get(1, null_on_oob=True).over("s1"))
                  .fill_null(0).alias(f"{c}_vs2nd")]
        cols += [c, f"{c}_rank", f"{c}_gap", f"{c}_vs2nd"]
    return df.with_columns(exprs), cols


def stack(data, work="/kaggle/working", tag=TAG):
    import lightgbm as lgb
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    os.makedirs(f"{work}/artifacts", exist_ok=True)
    sv, st = load_scores("val", work), load_scores("test", work)
    ovr = (glob.glob(f"{work}/stage03/fr_override_test.parquet")
           + glob.glob("/kaggle/input/**/stage03/fr_override_test.parquet", recursive=True))
    if ovr:
        o = pl.read_parquet(ovr[0])
        col = [c for c in o.columns if c.startswith("ce_")][0]
        st = (st.join(o.rename({col: "_ovr"}), on=["s1", "c"], how="left")
                .with_columns(pl.coalesce("_ovr", col).alias(col)).drop("_ovr"))
        print(f"French override applied: {o.height:,} pairs in column {col} (from {ovr[0]})")
    names = [c[3:] for c in sv.columns if c.startswith("ce_")]
    assert names == [c[3:] for c in st.columns if c.startswith("ce_")], "val/test scored with different models"
    fv, cols = features(sv, names)
    ft, _ = features(st, names)

    p_tr = find_checkpoint("train", work)
    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    val = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
             .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect()
             .filter(pl.col("fold") == VAL_FOLD).select("s1", "country")
             .with_columns(half=((pl.col("s1") // 10) % 2).cast(pl.Int8)))
    A = val.filter(pl.col("half") == 0).select("s1", "country")
    B = val.filter(pl.col("half") == 1).select("s1", "country")
    fv = (fv.join(gt.with_columns(y=pl.lit(1)), on=["s1", "c"], how="left")
            .with_columns(pl.col("y").fill_null(0)))
    gA = fv.join(A, on="s1", how="semi")
    es = (gA["s1"] // 10 % 7 == 0).to_numpy()          # 1/7 of half A for early stopping
    X = gA.select(cols).to_numpy()
    y = gA["y"].to_numpy()
    ceil_b = (gt.join(B, on="s1", how="semi")
                .join(fv.select("s1", "c"), on=["s1", "c"], how="semi").height
              / gt.join(B, on="s1", how="semi").height)
    print(f"stage-2 training pairs {int((~es).sum()):,} (pos {y[~es].mean():.3f}); "
          f"recall ceiling of top-{TOPN} on half B: {ceil_b:.4f}")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
    model = lgb.train(params, lgb.Dataset(X[~es], y[~es], feature_name=cols), 2000,
                      valid_sets=[lgb.Dataset(X[es], y[es], feature_name=cols)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    model.save_model(f"{work}/artifacts/stack_{tag}.txt")
    imp = pl.DataFrame({"feature": cols, "gain": model.feature_importance("gain")}).sort("gain", descending=True)
    print(imp.with_columns((pl.col("gain") / pl.col("gain").sum()).alias("share")))

    t04.section("HALF B: LightGBM alone vs stacked")
    pv_full = pl.read_parquet(find_path(f"**/artifacts/val_pred_{SRC_TAG}.parquet", work)).select("s1", "c", "p")
    gB = fv.join(B, on="s1", how="semi")
    pB = gB.select("s1", "c").with_columns(p=pl.Series(model.predict(gB.select(cols).to_numpy())).cast(pl.Float32))
    rows = []
    for label, pred in ((f"{SRC_TAG} LightGBM (all candidates)", pv_full.join(B, on="s1", how="semi")),
                        (f"{tag} stacked (top-{TOPN})", pB)):
        thr, rel, g = tune(pred, gt, B, thrs=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        r = g.row(0, named=True)
        rows.append({"source": label, "thr": thr, **{k: r[k] for k in ("F0.5", "P_avg", "R_avg", "F_singletons")}})
    res = pl.DataFrame(rows)
    print(res)
    print("reference: T17 blend on half B = 0.9828")
    best_thr = rows[1]["thr"]
    e = per_entity_f(decide(pB, best_thr), gt, B)
    print(summarize(e, by="country").select("country", "n", "F0.5", "P_avg", "R_avg", "F_singletons"))

    t04.section("APPLY TO TEST")
    pt = ft.select("s1", "c").with_columns(p=pl.Series(model.predict(ft.select(cols).to_numpy())).cast(pl.Float32))
    pt.write_parquet(f"{work}/artifacts/test_pred_{tag}.parquet")
    pred_t = decide(pt, best_thr)
    p_te = find_checkpoint("test", work)
    ids_te = pl.scan_parquet(p_te).select("id", "entity_id", "src", "country").collect()
    test_s1 = ids_te.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "entity_id")
    idmap = ids_te.filter(pl.col("src") != 1).select("id", "entity_id")
    out_dir = f"{work}/output_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(find_path(f"**/output_{SRC_TAG}/candidate_pairs.tsv", work), f"{out_dir}/candidate_pairs.tsv")
    t04.write_lists(test_s1, pred_t, idmap, "matched_entity_ids", f"{out_dir}/matching_results.tsv")
    cty = ids_te.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "country")
    print(cty.join(pred_t.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left")
             .with_columns(pl.col("k").fill_null(0))
             .group_by("country").agg(pl.len().alias("s1"), (pl.col("k") > 0).mean().alias("pred_nonempty"),
                                      pl.col("k").mean().alias("mean_pred")).sort("country"))
    print(f"wrote {out_dir}/ (threshold {best_thr}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "score":
        score(a[2], a[3], int(a[4]), int(a[5]))
    else:
        stack(a[2])
