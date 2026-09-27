# Business Entity Resolution: Reproduction Guide

This folder reproduces our two submission files from the competition data:

- `output/matching_results.tsv`: final matches (the leaderboard file)
- `output/candidate_pairs.tsv`: the exact candidate set the final models score

The method is described in `Documentation_template.md` (top of the submission zip).

---

## 1. Environment

- Python 3.11 or 3.12, Linux or Windows. We ran everything on Kaggle notebooks.
- `pip install -r requirements.txt`
- **CPU stages** (convert, normalize, block, tfidf, matcher): ~30 GB RAM recommended (peak ~21 GB).
- **GPU stages** (ce, blend): a CUDA GPU with PyTorch (`torch>=2.1`, platform CUDA build). We used 2 × Tesla T4 (16 GB). Cross-encoder training uses all visible GPUs automatically.
- Internet is needed once, to download the pretrained cross-encoder weights from Hugging Face (all Apache-2.0).

**No external data or lookup services are used.** Everything is learned from the provided files. The only built-in knowledge is in `src/ber/address.py` (state names/codes and common street abbreviations), written as plain tables in code.

## 2. Data layout

```
data/raw/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/raw/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 3. Running

Everything, in order:
```
python src/run_pipeline.py --stage all --raw data/raw --data data/parquet --work work
```

Or one stage at a time (each stage reads the checkpoints of the previous ones from `work/`):

| Stage | Command | Hardware | Time (Kaggle) | Writes |
|---|---|---|---|---|
| 1. convert | `--stage convert` | CPU | ~2 min | `data/parquet/*.parquet` |
| 2. normalize | `--stage normalize` | CPU | ~20 min | `work/stage01/`, `work/artifacts/` (native dictionaries, marker rules) |
| 3. block | `--stage block` | CPU | ~15 min | `work/stage02/cand_{split}_{country}.parquet` |
| 4. tfidf | `--stage tfidf` | CPU | ~1 h per shard, 7 shards | `work/stage02t/tf_*.parquet` |
| 5. matcher | `--stage matcher` | CPU | ~5.5 h | `work/artifacts/lgbm_t16.txt`, `val_pred_t16`, `test_pred_t16`, `work/output_t16/` |
| 6. ce | `--stage ce` | GPU | ~1-3 h per model | `work/ce_model_*` |
| 7a. blend (final if `FINAL = "blend"`) | `--stage blend` | GPU | ~1 h | `work/output_t17b/` → copied to `output/` |
| 7b. stack (final if `FINAL = "stack"`) | `--stage stack` | GPU + CPU | ~1.5 h per scoring shard, 10 min stacking | `work/output_t19/` → copied to `output/` |

The final submission files are written to `output/`. Which of 7a / 7b produces them is set by `FINAL` in `src/run_pipeline.py`.

**Parallel execution (what we actually did).** The TF-IDF search (stage 4) is independent per shard, so we ran its 7 shards in 7 parallel notebooks, and the cross-encoders (stage 6) in parallel GPU notebooks. The shard list and all model settings are in `src/run_pipeline.py` (`TFIDF_SHARDS`, `CROSS_ENCODERS`, `FRENCH_ADAPTATION`, `FINAL_BLEND`). Any single stage can be called directly, e.g.
```python
import t15b_tfidf_block
t15b_tfidf_block.main("data/parquet", "test", "India", part=0, n_parts=3, work="work")
```

## 4. What "reproduce" means here

- **Stages 1-5 are deterministic** (fixed seeds, fixed train/validation split derived from the S1 entity id).
- **Cross-encoder training (stage 6) is not bit-for-bit deterministic.** GPU arithmetic and data shuffling vary slightly between runs, so retrained models give equivalent but not identical scores.
- **The final stage (7) is exactly reproducible** from saved models and predictions: rerunning `--stage blend` on the same `work/` folder regenerates the same `output/` files.
- Some historical cross-encoders (`minilm`, `minilm12`) were trained on candidates from an earlier blocking version; retraining them with the current blocking gives models of the same kind and quality.

## 5. Code structure

```
src/
  run_pipeline.py          entry point (all stages, final configuration)
  stage00_convert_profile.py   TSV -> Parquet + data profile
  t06b_normalize.py        stage 2: normalization + learned dictionaries
  t07_blocking.py          stage 3: key-based blocking (5 passes)
  t15b_tfidf_block.py      stage 4: TF-IDF nearest-neighbour blocking (pass 6)
  t08_matcher.py           stage 5: features, LightGBM, thresholds, test inference
  t11a_ce_train.py         stage 6: cross-encoder fine-tuning
  t13_ce_france.py         stage 6: French adaptation with pseudo-labels
  t11b_ce_blend.py         stage 7a: cross-encoder rescoring, blend, final decisions
  t19_stack.py             stage 7b: cross-encoders on every top-10 pair + 2nd-stage model
  ber/                     shared library
    ids.py  split.py  metrics.py          ids, folds, exact F0.5 scorer
    normalize.py  address.py  stage01.py  cleaning and the stage-01 table
    blocking.py                           key-based blocking passes
    features.py  decide.py                pair features, decision rules
    ce.py                                 cross-encoder data/training/scoring
  t03_eda_pairs.py  t04_val_baselines.py  t05_test_inspect.py  t06_names.py
  t09_errors.py  t14a_sibling_probe.py  t15a_tfidf_probe.py  t_leak_check.py
                           analysis scripts used during development (not needed to reproduce)
```

## 6. Licences

Final models: LightGBM (MIT); cross-encoders fine-tuned from `cross-encoder/ms-marco-MiniLM-L-6-v2` and `cross-encoder/ms-marco-MiniLM-L-12-v2` (Apache-2.0; 22M and 33M parameters) and `microsoft/deberta-v3-small` (MIT; ~140M). All well within the 8B-parameter limit.
