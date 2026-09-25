"""T06b: data-driven name markers, address normalization v1, and the stage-01 checkpoint.

Kaggle usage (after bootstrap):
    import t06b_normalize
    t06b_normalize.main(DATA)
Writes /kaggle/working/stage01/{train,test}.parquet and /kaggle/working/artifacts/*.
"""
import gc
import json
import os
import sys
import time

import polars as pl

import t04_val_baselines as t04
from ber.ids import id_code
from ber.metrics import explode_ids, per_entity_f, summarize
from ber.normalize import B, clean_name_v1, learn_native_dict, num_ext
from ber.split import VAL_FOLD, fold_expr
from ber.address import apply_native_state, learn_native_state
from ber.stage01 import build_base, finalize

MARKERS = {"formerly": ["formerly known as", "formerly"], "fka": ["fka", "f/k/a"],
           "dba": ["dba", "d/b/a"], "aka": ["aka", "a/k/a"], "t/a": ["t/a"]}


def jacc(a, b):
    u = a.list.set_union(b).list.len()
    return pl.when(u > 0).then(a.list.set_intersection(b).list.len() / u)


def toks(e):
    return clean_name_v1(e, [], []).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def auc(df, score, label="y"):
    d = df.select(pl.col(score).cast(pl.Float64).fill_null(-1.0).rank("average").alias("r"), pl.col(label))
    npos = int(d[label].sum())
    nneg = d.height - npos
    if npos == 0 or nneg == 0:
        return None
    return (float(d.filter(pl.col(label))["r"].sum()) - npos * (npos + 1) / 2) / (npos * nneg)


# ---------------------------------------------------------------- 1. markers
def marker_analysis(data, gt):
    names = pl.concat([pl.read_parquet(f"{data}/train_source{s}.parquet",
                                       columns=["entity_id", "business_name"]) for s in "123"]
                      ).select(id=id_code(), name=pl.col("business_name"))
    s1_marked = (names.filter(pl.col("id") < 2 * 10_000_000_000)
                      .select(pl.col("name").str.to_lowercase()
                              .str.contains(rf"{B}(?:formerly|fka|dba|aka|t/a){B}").sum())).item()
    P = (gt.join(names.rename({"id": "c", "name": "c_name"}), on="c")
           .join(names.rename({"id": "s1", "name": "s1_name"}), on="s1")
           .with_columns(lc=pl.col("c_name").str.to_lowercase(), s1_t=toks(pl.col("s1_name"))))
    del names
    rows, after, before = [], [], []
    for label, pats in MARKERS.items():
        alt = "|".join(pats)
        sub = P.filter(pl.col("lc").str.contains(rf"{B}(?:{alt}){B}"))
        if sub.height == 0:
            continue
        sub = (sub.with_columns(bef=pl.col("lc").str.extract(rf"^(.*?){B}(?:{alt}){B}", 1),
                                aft=pl.col("lc").str.extract(rf"{B}(?:{alt}){B}[:\s]*(.*)$", 1))
                  .with_columns(jb=jacc(toks(pl.col("bef")), pl.col("s1_t")).fill_null(0.0),
                                ja=jacc(toks(pl.col("aft")), pl.col("s1_t")).fill_null(0.0)))
        share_a = float((sub["ja"] > sub["jb"]).mean())
        share_b = float((sub["jb"] > sub["ja"]).mean())
        keep = "after" if share_a > share_b else "before"
        (after if keep == "after" else before).extend(pats)
        rows.append({"marker": label, "true_pairs": sub.height,
                     "mean_j_before": round(float(sub["jb"].mean()), 3),
                     "mean_j_after": round(float(sub["ja"].mean()), 3),
                     "after_better": round(share_a, 3), "before_better": round(share_b, 3),
                     "keep": keep})
    print(pl.DataFrame(rows))
    print(f"S1 names containing a marker: {s1_marked:,}")
    # longest patterns first inside each list so 'formerly known as' wins over 'formerly'
    after = sorted(after, key=len, reverse=True)
    before = sorted(before, key=len, reverse=True)
    return after, before


# ---------------------------------------------------------------- helpers
def raw_for_ids(data, split, ids):
    """Raw name/address for a small set of ids (for printing samples)."""
    frames = []
    for s in "123":
        frames.append(pl.read_parquet(f"{data}/{split}_source{s}.parquet",
                                      columns=["entity_id", "business_name", "business_address"])
                        .with_columns(id=id_code()).filter(pl.col("id").is_in(ids)))
    return pl.concat(frames).drop("entity_id")


def print_samples(data, tr):
    ids = []
    for c in sorted(tr["country"].unique().to_list()):
        sub = tr.filter((pl.col("country") == c) & ~pl.col("addr_null"))
        ids += sub.sample(min(5, sub.height), seed=5)["id"].to_list()
    nat = tr.filter(pl.col("addr_native").list.len() > 0)
    ids += nat.sample(min(4, nat.height), seed=5)["id"].to_list()
    raw = raw_for_ids(data, "train", ids)
    print(tr.filter(pl.col("id").is_in(ids)).join(raw, on="id")
            .select("country", pl.col("business_address").str.slice(0, 70).alias("raw"), "state",
                    pl.col("addr_toks").list.join(" ").str.slice(0, 60).alias("addr_toks")))


# ---------------------------------------------------------------- address eval
def address_eval(data, feats, cand, gt, val):
    vc = cand.join(val, on="s1", how="semi").select("s1", "c", "num_ok")
    lab = (vc.join(gt.with_columns(y=pl.lit(True)), on=["s1", "c"], how="left")
             .with_columns(pl.col("y").fill_null(False)))
    need = pl.concat([lab.select(pl.col("s1").alias("id")), lab.select(pl.col("c").alias("id"))]).unique()
    raw = pl.concat([pl.read_parquet(f"{data}/train_source{s}.parquet",
                                     columns=["entity_id", "business_address"])
                       .with_columns(id=id_code()).join(need, on="id", how="semi")
                     for s in "123"])
    v0 = raw.select("id", v0_toks=pl.col("business_address").str.to_lowercase()
                    .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().str.split(" "))
    f = feats.join(need, on="id", how="semi").join(v0, on="id", how="left")
    a = f.rename({c: f"a_{c}" for c in f.columns})
    b = f.drop("country").rename({c: f"b_{c}" for c in f.columns if c != "country"})
    P = (lab.join(a, left_on="s1", right_on="a_id").join(b, left_on="c", right_on="b_id")
            .with_columns(addr_j_v0=jacc(pl.col("a_v0_toks"), pl.col("b_v0_toks")),
                          addr_j_v1=jacc(pl.col("a_addr_toks"), pl.col("b_addr_toks")),
                          num_trunc=(num_ext(pl.col("a_num")).list.set_intersection(num_ext(pl.col("b_num")))
                                     .list.len() > 0).fill_null(False),
                          state_same=(pl.col("a_state") == pl.col("b_state")),
                          state_missing=pl.col("a_state").is_null() | pl.col("b_state").is_null()))
    rows = []
    for country in sorted(P["a_country"].unique().to_list()):
        d = P.filter(pl.col("a_country") == country)
        pos, neg = d.filter(pl.col("y")), d.filter(~pl.col("y"))
        rows.append({
            "country": country, "pairs": d.height, "pos_rate": round(pos.height / d.height, 4),
            "auc_addr_v0": auc(d, "addr_j_v0"), "auc_addr_v1": auc(d, "addr_j_v1"),
            "auc_num_exact": auc(d, "num_ok"), "auc_num_trunc": auc(d, "num_trunc"),
            "pos_addr_v1_p50": pos["addr_j_v1"].median(), "neg_addr_v1_p50": neg["addr_j_v1"].median(),
            "pos_state_same": pos["state_same"].mean(), "neg_state_same": neg["state_same"].mean(),
            "state_missing": d["state_missing"].mean(),
        })
    print(pl.DataFrame(rows))


def state_coverage(df, tag):
    print(f"{tag}: state found share / top states")
    print(df.group_by("country", "src").agg(pl.len().alias("n"),
                                            pl.col("state").is_not_null().mean().alias("state_found"),
                                            pl.col("addr_toks").list.len().mean().alias("addr_toks_mean"))
            .sort("country", "src"))
    for c in sorted(df["country"].unique().to_list()):
        top = (df.filter(pl.col("country") == c).group_by("state").agg(pl.len().alias("n"))
                 .sort("n", descending=True).head(12))
        print(f"  {c}: " + "  ".join(f"{s}:{n}" for s, n in top.iter_rows()))


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_fmt_str_lengths(70)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    os.makedirs(f"{work}/stage01", exist_ok=True)
    os.makedirs(f"{work}/artifacts", exist_ok=True)

    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))

    t04.section("1. NAME MARKERS: which side matches S1?")
    after, before = marker_analysis(data, gt)
    json.dump({"after_markers": after, "before_markers": before},
              open(f"{work}/artifacts/name_markers_v1.json", "w"), indent=1)
    print(f"keep text AFTER: {after}\nkeep text BEFORE: {before}  {t04.mem()}")
    gc.collect()

    t04.section("2. BUILD + SAVE TRAIN STAGE-01")
    base = build_base(data, "train", after, before)
    s1 = (base.filter(pl.col("src") == 1)
              .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")))
    val = s1.filter(pl.col("fold") == VAL_FOLD).select("s1", "country")
    gt_learn = gt.join(s1.filter(pl.col("fold") != VAL_FOLD).select("s1"), on="s1", how="semi")
    d, n_al = learn_native_dict(base.select("id", pl.col("name_toks").alias("toks"), "nonlatin"), gt_learn)
    d.write_parquet(f"{work}/artifacts/native_dict_v1.parquet")
    print(f"native name dictionary: {d.height:,} entries from {n_al:,} aligned pairs (folds != {VAL_FOLD})")
    tr, stats = finalize(base, d)
    del base
    gc.collect()
    print(f"train native name coverage: {stats}  {t04.mem()}")
    st_map = learn_native_state(tr.select("id", "src", "state", "addr_native"), gt_learn)
    st_map.write_parquet(f"{work}/artifacts/native_state_v1.parquet")
    print(f"native state map: {st_map.height} tokens; top: "
          + "  ".join(f"{t}->{s}" for t, s in st_map.select("token", "state").head(12).iter_rows()))
    tr = apply_native_state(tr, st_map)
    print_samples(data, tr)
    state_coverage(tr, "TRAIN")
    tr.write_parquet(f"{work}/stage01/train.parquet", compression="zstd")
    print(f"wrote {work}/stage01/train.parquet ({os.path.getsize(f'{work}/stage01/train.parquet') / 1e6:.0f} MB)")
    del tr
    gc.collect()

    t04.section("3. BUILD + SAVE TEST STAGE-01")
    base_t = build_base(data, "test", after, before)
    te, stats_t = finalize(base_t, d)
    del base_t
    gc.collect()
    te = apply_native_state(te, st_map)
    print(f"TEST native name coverage: {stats_t}")
    state_coverage(te, "TEST")
    te.write_parquet(f"{work}/stage01/test.parquet", compression="zstd")
    print(f"wrote {work}/stage01/test.parquet ({os.path.getsize(f'{work}/stage01/test.parquet') / 1e6:.0f} MB)"
          f"  {t04.mem()}")
    del te
    gc.collect()

    t04.section("4. B3 REPRODUCTION WITH STAGE-01 KEYS (validation)")
    keys = pl.read_parquet(f"{work}/stage01/train.parquet",
                           columns=["id", "src", "country", "k1", "k2", "num", "addr_null"])
    cand, _ = t04.key_candidates(keys)
    del keys
    gc.collect()
    pred = t04.baselines(cand)["B3_namekey_num_excl"]
    e = per_entity_f(pred, gt, val)
    vc = cand.join(val, on="s1", how="semi")
    gv = gt.join(val, on="s1", how="semi")
    ceil = gv.join(vc.select("s1", "c"), on=["s1", "c"], how="semi").height / gv.height
    print(pl.concat([summarize(e).with_columns(country=pl.lit("ALL")), summarize(e, by="country")],
                    how="diagonal_relaxed").select("country", "n", "F0.5", "P_avg", "R_avg",
                                                   "pred_nonempty", "F_singletons"))
    print(f"recall ceiling {ceil:.4f}; {vc.height / val.height:.1f} candidates per S1 "
          f"(T06 C2 reference: F0.5 0.6452, ceiling 0.6329)  {t04.mem()}")
    del pred, e, vc, gv
    gc.collect()

    t04.section("5. ADDRESS SIGNALS ON NAME-KEY CANDIDATES (validation, AUC)")
    feats = pl.read_parquet(f"{work}/stage01/train.parquet",
                            columns=["id", "country", "addr_toks", "state", "num"])
    address_eval(data, feats, cand, gt, val)
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet", sys.argv[2] if len(sys.argv) > 2 else ".")
