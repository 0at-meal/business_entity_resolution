"""T06: name normalization v1 + native-script dictionary + truncation-tolerant numbers.
Each change is measured with the B3 heuristic on the validation fold.

Kaggle usage (after bootstrap, fresh kernel):
    import importlib, t06_names
    importlib.reload(t06_names)
    t06_names.main(DATA)
"""
import gc
import os
import sys
import time

import polars as pl

import t04_val_baselines as t04
from ber.ids import id_code
from ber.metrics import explode_ids, per_entity_f, summarize
from ber.normalize import (addr_numbers_int, apply_native_dict, core_from_tokens_v1, is_nonlatin,
                           learn_native_dict, name_core_tokens, name_tokens_v1, num_ext)
from ber.split import VAL_FOLD, fold_expr

CONFIGS = {  # name: (k1 column, k2 column, number column)
    "C0_v0": ("k1_v0", "k2_v0", "num"),
    "C1_v1names": ("k1_v1", "k2_v1", "num"),
    "C2_v1+native": ("k1_v2", "k2_v2", "num"),
    "C3_v1+native+numext": ("k1_v2", "k2_v2", "num_ext"),
}


def load(data, split):
    frames = []
    for s in ("1", "2", "3"):
        df = pl.read_parquet(f"{data}/{split}_source{s}.parquet",
                             columns=["entity_id", "business_name", "business_address", "country"])
        df = (df.select(id=id_code(),
                        entity_id=pl.col("entity_id"),
                        src=pl.lit(int(s), pl.Int8),
                        country=pl.col("country"),
                        name=pl.col("business_name"),
                        core0=name_core_tokens(pl.col("business_name")),
                        toks=name_tokens_v1(pl.col("business_name")),
                        nonlatin=is_nonlatin(pl.col("business_name")).fill_null(False),
                        num=addr_numbers_int(pl.col("business_address")),
                        addr_null=pl.col("business_address").is_null())
                .with_columns(k1_v0=pl.col("core0").list.join(""),
                              k2_v0=pl.col("core0").list.sort().list.join(" "))
                .drop("core0"))
        frames.append(df)
        del df
        gc.collect()
    return pl.concat(frames)


def add_keys(recs, toks_col, tag):
    return (recs.with_columns(_core=core_from_tokens_v1(pl.col(toks_col)))
                .with_columns(pl.col("_core").list.join("").alias(f"k1_{tag}"),
                              pl.col("_core").list.sort().list.join(" ").alias(f"k2_{tag}"))
                .drop("_core"))


def view(recs, cfg):
    k1, k2, num = CONFIGS[cfg]
    return recs.select("id", "src", "country", pl.col(k1).alias("k1"), pl.col(k2).alias("k2"),
                       pl.col(num).alias("num"), "addr_null")


def evaluate(cfg, recs, gt, val):
    cand, _ = t04.key_candidates(view(recs, cfg))
    vc = cand.join(val, on="s1", how="semi")
    gv = gt.join(val, on="s1", how="semi")
    hit = gv.join(vc.select("s1", "c"), on=["s1", "c"], how="semi")
    per_c = (val.group_by("country").agg(pl.len().alias("n_s1"))
                .join(gv.join(val, on="s1").group_by("country").agg(pl.len().alias("tp")), on="country")
                .join(hit.join(val, on="s1").group_by("country").agg(pl.len().alias("hit")),
                      on="country", how="left")
                .join(vc.join(val, on="s1").group_by("country").agg(pl.len().alias("nc")),
                      on="country", how="left")
                .with_columns(ceiling=pl.col("hit") / pl.col("tp"),
                              cands_per_s1=pl.col("nc") / pl.col("n_s1"))
                .select("country", "ceiling", "cands_per_s1"))
    all_c = pl.DataFrame({"country": ["ALL"], "ceiling": [hit.height / gv.height],
                          "cands_per_s1": [vc.height / val.height]})
    pred = t04.baselines(cand)["B3_namekey_num_excl"]
    e = per_entity_f(pred, gt, val)
    s = pl.concat([summarize(e).with_columns(country=pl.lit("ALL")),
                   summarize(e, by="country")], how="diagonal_relaxed")
    out = (s.join(pl.concat([all_c, per_c]), on="country", how="left")
            .with_columns(config=pl.lit(cfg))
            .select("config", "country", "ceiling", "cands_per_s1", "F0.5", "P_avg", "R_avg",
                    "pred_nonempty", "F_singletons"))
    del cand, vc, pred, e
    gc.collect()
    return out


def show_samples(recs):
    lname = pl.col("name").str.to_lowercase()
    picks = [
        ("native", pl.col("nonlatin")),
        ("domain", lname.str.contains(r"\.com\b|^@")),
        ("formerly", lname.str.contains(r"\bformerly\b")),
        ("legal+generic", lname.str.contains(r"\b(?:ltd|llc|inc|limited)\b\W+(?:center|services|partners)\b")),
        ("spaced legal", lname.str.contains(r"\b(?:l l c|p c|s a s|s a r l)\b")),
    ]
    rows = []
    for label, cond in picks:
        sub = recs.filter(cond)
        if sub.height:
            rows.append(sub.sample(min(4, sub.height), seed=3)
                           .select(pl.lit(label).alias("case"), "name",
                                   pl.col("toks_nat").list.join(" ").alias("tokens_after"),
                                   pl.col("k1_v2").alias("key_v1")))
    print(pl.concat(rows))
    lname_all = recs.select(lname.alias("n"))
    print("marker counts:",
          {m: int(lname_all["n"].str.contains(rf"\b{m}\b").sum())
           for m in ["formerly", "fka", "dba", "aka", "t/a"]})


def main(data, work="/kaggle/working"):
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_float_precision(4)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    t0 = time.time()
    os.makedirs(f"{work}/artifacts", exist_ok=True)

    t04.section("1. LOAD TRAIN + v1 KEYS")
    tr = load(data, "train")
    gt = (explode_ids(pl.read_parquet(f"{data}/train_ground_truth.parquet"),
                      "source1_entity_id", "matched_entity_ids")
          .select(s1=id_code("s1"), c=id_code("c")))
    s1 = (tr.filter(pl.col("src") == 1)
            .select(pl.col("id").alias("s1"), "country", fold_expr("entity_id")))
    val = s1.filter(pl.col("fold") == VAL_FOLD).select("s1", "country")
    gt_learn = gt.join(s1.filter(pl.col("fold") != VAL_FOLD).select("s1"), on="s1", how="semi")
    tr = add_keys(tr, "toks", "v1")
    print(f"train loaded {t04.mem()} in {time.time() - t0:.0f}s")

    t04.section("2. NATIVE-SCRIPT DICTIONARY (learned on folds != val)")
    d_val, n_aligned = learn_native_dict(tr.select("id", "toks", "nonlatin"), gt_learn)
    print(f"aligned native/S1 pairs used: {n_aligned:,}; dictionary entries: {d_val.height:,}")
    print(d_val.head(25))
    tok_frame, stats = apply_native_dict(tr.select("id", "toks", "nonlatin"), d_val)
    print(f"train native coverage: {stats}")
    tr = (tr.join(tok_frame.select("id", "toks_nat"), on="id", how="left")
            .with_columns(num_ext=num_ext(pl.col("num"))))
    tr = add_keys(tr, "toks_nat", "v2")
    del tok_frame
    gc.collect()

    t04.section("3. BEFORE / AFTER SAMPLES")
    show_samples(tr)
    tr = tr.drop("name", "entity_id", "toks")
    gc.collect()

    t04.section("4. CONFIG COMPARISON ON VALIDATION (B3 heuristic)")
    results = []
    for cfg in CONFIGS:
        t1 = time.time()
        results.append(evaluate(cfg, tr, gt, val))
        print(f"  {cfg} done in {time.time() - t1:.0f}s {t04.mem()}", flush=True)
    res = pl.concat(results)
    print(res.sort("country", "config"))
    best = (res.filter(pl.col("country") == "ALL").sort("F0.5", descending=True)["config"][0])
    print(f"best config on validation: {best}")

    t04.section("5. DICTIONARY ON ALL TRAIN -> TEST")
    del tr
    gc.collect()
    tr2 = load(data, "train").select("id", "toks", "nonlatin")
    d_all, n_all = learn_native_dict(tr2, gt)
    d_all.write_parquet(f"{work}/artifacts/native_dict_v1.parquet")
    print(f"full-train dictionary: {d_all.height:,} entries from {n_all:,} aligned pairs "
          f"-> {work}/artifacts/native_dict_v1.parquet")
    del tr2
    gc.collect()

    te = load(data, "test")
    te = add_keys(te, "toks", "v1")
    tok_frame, stats_t = apply_native_dict(te.select("id", "toks", "nonlatin"), d_all)
    print(f"TEST native coverage: {stats_t}")
    te = (te.join(tok_frame.select("id", "toks_nat"), on="id", how="left")
            .with_columns(num_ext=num_ext(pl.col("num"))))
    te = add_keys(te, "toks_nat", "v2")
    del tok_frame
    gc.collect()

    test_s1 = te.filter(pl.col("src") == 1).select(pl.col("id").alias("s1"), "entity_id", "country")
    idmap = te.filter(pl.col("src") != 1).select("id", "entity_id")
    cand_t, _ = t04.key_candidates(view(te, best))
    del te
    gc.collect()
    pred_t = t04.baselines(cand_t)["B3_namekey_num_excl"]
    out_dir = f"{work}/output_t06"
    os.makedirs(out_dir, exist_ok=True)
    t04.write_lists(test_s1.select("s1", "entity_id"), cand_t.select("s1", "c"), idmap,
                    "candidate_entity_ids", f"{out_dir}/candidate_pairs.tsv")
    t04.write_lists(test_s1.select("s1", "entity_id"), pred_t, idmap,
                    "matched_entity_ids", f"{out_dir}/matching_results.tsv")
    print(test_s1.join(pred_t.group_by("s1").agg(pl.len().alias("k")), on="s1", how="left")
                 .with_columns(pl.col("k").fill_null(0))
                 .group_by("country").agg(pl.len().alias("s1"),
                                          (pl.col("k") > 0).mean().alias("pred_nonempty"),
                                          pl.col("k").mean().alias("mean_pred"))
                 .sort("country"))
    print(f"wrote {out_dir}/ using {best}")
    print(f"\ndone in {time.time() - t0:.0f}s {t04.mem()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/parquet",
         sys.argv[2] if len(sys.argv) > 2 else ".")
