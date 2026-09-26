"""T11b: blend the cross-encoder (T11a) into the T10 LightGBM probabilities.

Only UNCERTAIN pairs (LO <= p <= HI) are re-scored; confident pairs keep the LightGBM probability.
Validation S1s are split in two halves: the blend is fitted on half A and compared against T10
alone on half B (decision rules re-tuned for both), so the comparison is fair.

Needs: stage01 (ber-00), T10 artifacts + output_t10 (ber-01), ce_model_minilm (ber-03), GPU.
Kaggle usage (after bootstrap):
    import t11b_ce_blend
    t11b_ce_blend.main(DATA)
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
import torch

import t04_val_baselines as t04
from ber.ce import TEXT_COLS, attach_text, predict
from ber.decide import decide, decide_expected, tune, tune_expected
from ber.ids import id_code
from ber.metrics import explode_ids, per_entity_f, summarize
from ber.split import VAL_FOLD, fold_expr
from ber.stage01 import find_checkpoint

SRC_TAG = "t10"
TAG = "t13"
CE_NAMES = ["minilm", "minilm12"]   # every cross-encoder found is used; missing ones are skipped
LO, HI = 0.02, 0.999                # LO matches the lowest test probability saved by T10 (KEEP_P)
FR_OVERRIDE = {"minilm12": "minilm12_fr"}  # for French test pairs, the French-adapted model fills this slot
MAXLEN = 96


def find_path(pattern, work):
    hits = glob.glob(f"{work}/{pattern}", recursive=True) + glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found in {work} or /kaggle/input (attach the right notebook output)")
    return hits[0]


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def ce_score(pairs, stage_path, models):
    """Score pairs with every cross-encoder in `models` ({name: (model, tok)}); adds ce_<name>."""
    ids = pl.concat([pairs.select(pl.col("s1").alias("id")), pairs.select(pl.col("c").alias("id"))]).unique()
    text = pl.scan_parquet(stage_path).select(TEXT_COLS).collect().join(ids, on="id", how="semi")
    df = attach_text(pairs, text)
    out = df.select("s1", "c")
    a, b = df["a_text"].to_list(), df["b_text"].to_list()
    for name, (model, tok) in models.items():
        t1 = time.time()
        s = predict(model, tok, a, b, MAXLEN)
        out = out.with_columns(pl.Series(f"ce_{name}", s, dtype=pl.Float32))
        print(f"  {name}: {len(a):,} pairs in {time.time() - t1:.0f}s", flush=True)
    return out


def blend_features(df, names):
    lp = logit(df["p"].to_numpy())
    cols = [lp]
    for n in names:
        ce = df[f"ce_{n}"].to_numpy().astype(np.float64)
        cols += [ce, lp * ce]
    return np.column_stack(cols)


def best_rule(pred, gt, base, label):
    thr, rel, g1 = tune(pred, gt, base)
    alpha, floor, g2 = tune_expected(pred, gt, base)
    r1, r2 = g1.row(0, named=True), g2.row(0, named=True)
    if r2["F0.5"] > r1["F0.5"]:
        return {"source": label, "rule": "expected", "a": alpha, "b": floor, **{k: r2[k] for k in ("F0.5", "P_avg", "R_avg", "F_singletons")}}
    return {"source": label, "rule": "threshold", "a": thr, "b": rel, **{k: r1[k] for k in ("F0.5", "P_avg", "R_avg", "F_singletons")}}


def apply_rule(pred, r):
    return decide_expected(pred, r["a"], r["b"]) if r["rule"] == "expected" else decide(pred, r["a"], r["b"])


def main(data, work="/kaggle/working"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU"
    os.makedirs(f"{work}/artifacts", exist_ok=True)

    p_tr, p_te = find_checkpoint("train", work), find_checkpoint("test", work)
    pv = pl.read_parquet(find_path(f"**/artifacts/val_pred_{SRC_TAG}.parquet", work))
    pt = pl.read_parquet(find_path(f"**/artifacts/test_pred_{SRC_TAG}.parquet", work))
    cand_file = find_path(f"**/output_{SRC_TAG}/candidate_pairs.tsv", work)
    models = {}
    for n in CE_NAMES:
        try:
            d = os.path.dirname(find_path(f"**/ce_model_{n}/config.json", work))
        except FileNotFoundError:
            print(f"cross-encoder '{n}' not found - skipped")
            continue
        models[n] = (AutoModelForSequenceClassification.from_pretrained(d).to("cuda").half(),
                     AutoTokenizer.from_pretrained(d))
        print(f"loaded cross-encoder '{n}' from {d}")
    assert models, "no cross-encoder found: attach the ber-03 output(s)"
    models_fr = {}
    for base_name, fr_name in FR_OVERRIDE.items():
        if base_name not in models:
            continue
        try:
            d = os.path.dirname(find_path(f"**/ce_model_{fr_name}/config.json", work))
        except FileNotFoundError:
            print(f"French override '{fr_name}' not found - France uses '{base_name}'")
            continue
        models_fr[base_name] = (AutoModelForSequenceClassification.from_pretrained(d).to("cuda").half(),
                                AutoTokenizer.from_pretrained(d))
        print(f"French pairs: '{fr_name}' replaces '{base_name}' (from {d})")
    names = list(models)
    print(f"val preds {pv.height:,} | test preds {pt.height:,} | cross-encoders {names}")

    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    val = (pl.scan_parquet(p_tr).filter(pl.col("src") == 1)
             .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")).collect()
             .filter(pl.col("fold") == VAL_FOLD).select("s1", "country")
             .with_columns(half=((pl.col("s1") // 10) % 2).cast(pl.Int8)))   # val ids all end in 0
    A = val.filter(pl.col("half") == 0).select("s1", "country")
    B = val.filter(pl.col("half") == 1).select("s1", "country")

    t04.section("1. SCORE UNCERTAIN VALIDATION PAIRS")
    uv = pv.filter(pl.col("p").is_between(LO, HI))
    print(f"uncertain val pairs: {uv.height:,} of {pv.height:,} ({uv.height / pv.height:.3f})")
    t1 = time.time()
    uv = uv.join(ce_score(uv.select("s1", "c"), p_tr, models), on=["s1", "c"])
    uv = (uv.join(gt.with_columns(y=pl.lit(1)), on=["s1", "c"], how="left")
            .with_columns(pl.col("y").fill_null(0)))
    print(f"scored in {time.time() - t1:.0f}s")

    t04.section("2. FIT BLEND ON HALF A, COMPARE ON HALF B")
    ua = uv.join(A, on="s1", how="semi")
    ub = uv.join(B, on="s1", how="semi")
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(blend_features(ua, names), ua["y"].to_numpy())
    print(f"blend coefficients [logit_p, (ce, logit_p*ce) per model {names}]: "
          f"{np.round(lr.coef_[0], 4)} intercept {lr.intercept_[0]:.4f}")
    pb = lr.predict_proba(blend_features(ub, names))[:, 1]
    yb = ub["y"].to_numpy()
    aucs = " | ".join(f"{n} {roc_auc_score(yb, ub[f'ce_{n}'].to_numpy()):.5f}" for n in names)
    print(f"half-B uncertain pairs {ub.height:,} (pos {yb.mean():.3f}) AUC: "
          f"LightGBM {roc_auc_score(yb, ub['p'].to_numpy()):.5f} | {aucs} | blend {roc_auc_score(yb, pb):.5f}")
    uv = uv.with_columns(p_blend=pl.Series(lr.predict_proba(blend_features(uv, names))[:, 1], dtype=pl.Float32))
    pv2 = (pv.join(uv.select("s1", "c", "p_blend"), on=["s1", "c"], how="left")
             .with_columns(p=pl.coalesce("p_blend", "p")).drop("p_blend"))
    res = [best_rule(pv.join(B, on="s1", how="semi"), gt, B, f"{SRC_TAG} LightGBM"),
           best_rule(pv2.join(B, on="s1", how="semi"), gt, B, f"{TAG} blend")]
    print(pl.DataFrame(res))
    win = res[1] if res[1]["F0.5"] > res[0]["F0.5"] else res[0]
    print(f"winner on half B: {win['source']} with {win['rule']} rule")
    if win is res[0]:
        print("The blend did not beat T10 on held-out validation: keep output_t10 as the submission.")
        print(f"done in {time.time() - t0:.0f}s")
        return

    t04.section("3. APPLY TO TEST")
    ut = pt.filter(pl.col("p").is_between(LO, HI))
    print(f"uncertain test pairs: {ut.height:,} of {pt.height:,}")
    t1 = time.time()
    cty_map = (pl.scan_parquet(p_te).filter(pl.col("src") == 1)
                 .select(pl.col("id").alias("s1"), "country").collect())
    ut = ut.join(cty_map, on="s1")
    parts = []
    for is_fr, sub in ((True, ut.filter(pl.col("country") == "France")),
                       (False, ut.filter(pl.col("country") != "France"))):
        if sub.height == 0:
            continue
        mdl = {**models, **models_fr} if is_fr else models
        print(f"  {'France' if is_fr else 'other countries'}: {sub.height:,} pairs", flush=True)
        parts.append(sub.join(ce_score(sub.select("s1", "c"), p_te, mdl), on=["s1", "c"]))
    ut = pl.concat(parts).drop("country")
    print(f"scored in {time.time() - t1:.0f}s")
    ut = ut.with_columns(p_blend=pl.Series(lr.predict_proba(blend_features(ut, names))[:, 1], dtype=pl.Float32))
    pt2 = (pt.join(ut.select("s1", "c", "p_blend"), on=["s1", "c"], how="left")
             .with_columns(p=pl.coalesce("p_blend", "p")).drop("p_blend"))
    pt2.write_parquet(f"{work}/artifacts/test_pred_{TAG}.parquet")
    pred_t = apply_rule(pt2, win)

    ids_te = pl.scan_parquet(p_te).select("id", "entity_id", "src", "country").collect()
    test_s1 = ids_te.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "entity_id")
    idmap = ids_te.filter(pl.col("src") != 1).select("id", "entity_id")
    out_dir = f"{work}/output_{TAG}"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cand_file, f"{out_dir}/candidate_pairs.tsv")
    t04.write_lists(test_s1, pred_t, idmap, "matched_entity_ids", f"{out_dir}/matching_results.tsv")
    cty = ids_te.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "country")
    print(cty.join(pred_t.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left")
             .with_columns(pl.col("k").fill_null(0))
             .group_by("country").agg(pl.len().alias("s1"), (pl.col("k") > 0).mean().alias("pred_nonempty"),
                                      pl.col("k").mean().alias("mean_pred")).sort("country"))
    print(f"wrote {out_dir}/matching_results.tsv and candidate_pairs.tsv")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
