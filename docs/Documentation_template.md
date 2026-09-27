# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We built a four-stage pipeline: country-aware normalization with dictionaries learned from the training pairs, a six-pass blocking stage that combines rare-token keys with character 3-gram TF-IDF nearest-neighbour search (raising the recall ceiling from 0.65 to about 0.99), a LightGBM matcher whose key features describe how strongly each S1 entity competes for a record, and fine-tuned transformer cross-encoders (MiniLM, DeBERTa-v3) that rescore every business's top-10 candidates, combined with LightGBM by a second-stage model. The main innovations are (1) "competition" features that let the model learn the one-owner-per-record structure instead of hard-coding it, (2) TF-IDF blocking to recover heavily corrupted copies that no exact key reaches, and (3) pseudo-label adaptation of the cross-encoder to France, which has no training labels.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from exploring the training data (all measured, not assumed):

- **One owner per record.** No S2/S3 record is matched to more than one S1 entity (0 of 7.6M true pairs). Country always agrees within a pair (100%).
- **Few singletons, many matches.** 5.6% of S1 entities have no match; the rest have 3.5 matches on average.
- **Name noise:** typos, digit-for-letter swaps (`G1obal`, `8lue`), injected accents, word reordering, legal-form churn (`Pvt Ltd` ↔ `Private Limited` ↔ dropped), prefixes (`Dr`, `M/s`), generic words appended after the legal form (`Coex Ltd Services Services`), names turned into web domains (`holleybrands.com`), and invented brands joined to the real name by a marker (`Nexbelo DBA Psv Education Ltd`). Measured on 126k true pairs: the text **after** `formerly/fka/dba/aka/t/a` matches S1 in 99.8–100% of cases.
- **Native scripts.** About 23% of Indian S2 names and 13% of S3 names are written in Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Gurmukhi or Oriya: word-by-word transliterations of the English name (551k true pairs share zero letters with S1).
- **Address noise:** abbreviations, state names vs codes (including native-script state names), reordered components, city changed or dropped (city is unreliable; street and locality are reliable), house numbers truncated (`257` → `57`), zero-padded or prefixed, 3% of S2/S3 addresses missing. There are effectively no postal codes (0% of Indian PINs).
- **Deliberate lookalikes.** About 26% of S2/S3 records match nothing; many are copies of a real business with one meaningful word swapped (`Randle and Kemp Kochav LLC` vs `SANDERS AND KEMP KOCHAV LLC`) or a different house number.
- **France (test only)** was produced by the same noise generator (checked on pseudo-labelled test clusters), plus French specifics: `St` = Saint, spaced legal forms (`S A S`), region ↔ department swaps, and 35% of S2/S3 records without a region.
- **No leakage:** entity ids and file row order are uncorrelated with matches (Spearman ≈ 0.0002, identical to random pairs).

### 2.2 Solution Strategy

**Approach Type:** Hybrid: multi-pass blocking + gradient-boosted classifier + fine-tuned cross-encoder reranker, with assignment-based decisions.

**Core Innovation:** Treating "which S1 entity owns this record" as evidence for the model (rank and score gap among all S1 entities competing for the same record) rather than a hard rule, combined with TF-IDF character-level blocking and cross-encoders that read both records together.

The pipeline:

1. **Normalize** names and addresses; learn a native-script → Latin token dictionary (1,347 entries, 96–98% coverage) and native state names (16 tokens) from aligned true pairs.
2. **Block** with five key-based passes plus TF-IDF nearest neighbours.
3. **Pre-filter** with a cheap similarity score to the top 50 candidates per S1, always keeping TF-IDF pairs.
4. **Score** each pair with LightGBM on 45 features.
5. **Rescore** each S1's top-10 candidates (by LightGBM probability) with two cross-encoders (MiniLM-L12 and DeBERTa-v3-small; a French-adapted MiniLM-L12 for French pairs).
6. **Stack:** a second-stage LightGBM combines the LightGBM logit, the cross-encoder logits and their within-S1 ranks and gaps.
7. **Decide:** threshold 0.7, then each S2/S3 record goes to at most one S1 (the highest probability).

All decisions were validated on a held-out fold (10% of training S1 entities, split by entity id) with an exact reimplementation of the macro F0.5 metric.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used** (all within country; each key hashed to 64 bits; keys shared by more than a cap of S2/S3 records are skipped):
  1. `name_exact`: normalized name core, glued (k1) and word-sorted (k2), with digit-for-letter fixes (cap 300)
  2. `num_name`: house number (+ leading-digit-truncated variant) + each of the record's 2 rarest name words (cap 50)
  3. `num_addr`: house number (+ truncated variant) + each of the 2 rarest address words (cap 50): catches renamed businesses
  4. `name_pair`: every pair among the 3 rarest name words (cap 50): reordering, added/dropped words, missing addresses
  5. `name_prefix`: state + first 3 letters of the first two name words (cap 50): late typos
  6. **TF-IDF**: character 3-gram TF-IDF (hashed, sublinear tf, L2-normalized) on "name core + address"; top-20 cosine neighbours per S1 within its state (cosine ≥ 0.3), computed with sparse top-n matrix multiplication

  "Rarest" words are counted within the country and must appear at least twice (typo-made words appear once and can never match).
- **Candidate pairs generated (test):** 164.3M from the key passes, plus 23.2M new pairs from TF-IDF (187.5M total). After the cheap pre-filter, **83.7M pairs** are scored by the models; this set is `candidate_pairs.tsv`.
- **How we ensured true matches were not lost:** every pass was added to recover a measured failure mode, and each was evaluated on validation by its marginal recall:

| After pass | Recall ceiling (validation) |
|---|---|
| name_exact | 0.666 |
| + num_name | 0.871 |
| + num_addr | 0.935 |
| + name_pair | 0.954 |
| + name_prefix | 0.961 |
| + TF-IDF top-20 | ~0.987 |

  Coverage (S1 entities with at least one true candidate) is 99.6% before TF-IDF. After the top-50 pre-filter, the ceiling is still **0.982 (India) and 0.989 (US)**, because TF-IDF pairs bypass the cut and a missing address no longer penalizes the cheap score.

---

## 4. Matching Model

**Features used:**
- **Name features:** Levenshtein ratio, Jaro-Winkler, token sort / token set / partial ratio (rapidfuzz), core-token Jaccard and containment, exact-core flag, core lengths, and **residual-word similarity** (edit ratio between the words each side has that the other lacks, which separates typos from word-swap lookalikes).
- **Address features:** address-token Jaccard and containment, character ratio, residual address-word similarity, missing-address flags, state agreement (same / different / unknown).
- **House numbers:** shared-number count, leading-digit-truncated match, number counts, number-conflict flag.
- **Competition (context) features:** rank and score gap of the pair among (a) the S1 entity's candidates and (b) **all S1 entities competing for the same record**, plus candidate counts. These account for about 75% of the model's total gain.
- **Other:** candidate source (S2/S3), native-script flag, blocking-pass bits, TF-IDF cosine and rank.
- **Country is deliberately not a feature**, so the model transfers to France.

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.1, early stopping on validation logloss; 2,259 trees), trained on 34M pairs from 35% of training S1 entities. On top of it, fine-tuned cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2` and `-L-12-v2`, Apache-2.0, 22M/33M parameters) read `name | address | state` for both records together. They were trained on each S1's 10 hardest candidates (up to 5.9M pairs, 2 epochs, mixed precision, 2 × T4 GPUs; best validation AUC 0.99956). A second architecture, `microsoft/deberta-v3-small` (MIT, ~140M parameters), reached AUC 0.99953 with half the training data and adds diversity. **Final configuration (T19):** both cross-encoders score every S1's top-10 candidates by LightGBM probability (recall ceiling of the top-10 set: 0.985), and a second-stage LightGBM, trained on one half of the validation entities, combines the LightGBM logit, both cross-encoder logits, and each score's rank, gap to the best and margin over the second best within the S1. For France, which has no labels, the MiniLM-L12 cross-encoder was further trained on high-confidence pseudo-labels (p ≥ 0.98 / < 0.02) mixed with original pairs, and its scores replace the original model's for French pairs only. (An earlier configuration rescored only uncertain pairs, probability 0.02–0.999, and combined scores with a logistic-regression blend; it scored 0.976.)

**Threshold selection method:** grid search of the probability threshold (0.2–0.9) maximizing macro F0.5 on validation (chosen: 0.7), followed by one-owner-per-record assignment. An expected-F0.5 per-entity rule was also evaluated and lost narrowly (0.9729 vs 0.9736), mainly on singletons. The blend was fitted on one half of the validation entities and evaluated on the other half.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9829** on held-out validation (final stacked model, evaluated on validation entities not used to fit it); LightGBM alone 0.9736. Precision 0.994, recall 0.962, singletons 0.989. **Leaderboard: 0.978.**

| Version | Main change | Validation | Leaderboard |
|---|---|---|---|
| B3 rule | exact name + house number | 0.615 | — |
| T08 | LightGBM matcher | 0.958 | 0.950 |
| T10 | blocking v3, residual features | 0.966 | 0.956 |
| T11 | + cross-encoder blend | 0.973 | 0.966 |
| T13 | + larger cross-encoder, French adaptation | 0.974 | 0.969 |
| T17 | + TF-IDF blocking, stronger cross-encoder (French model wired into a negatively weighted slot) | 0.983 | 0.972 |
| T17b | same, French override removed | 0.983 | 0.976 |
| **T19 (final)** | **cross-encoders on every top-10 pair + DeBERTa + French MiniLM-L12, second-stage LightGBM** | **0.983** | **0.978** |

- **Common false positives (wrong merges):**
  - **Deliberate lookalikes** (68% of false matches in our error analysis): one meaningful name word swapped at the same address (`Alvarez Sky Group` vs `SMITH SKY GROUP`), or the same name at a different house number. Residual-word features and the cross-encoders cut these substantially.
  - **Address-less records with identical names**, assigned to the wrong one of several same-name S1 entities.
  - **Singletons** receiving a lookalike (461 of 697,754 predicted pairs in our first model's error analysis).
- **Common false negatives (missed matches):**
  - **Records corrupted in both name and address at once** (typo in the name *and* a missing or truncated house number), which no exact key reaches. TF-IDF blocking recovered about two-thirds of these.
  - **Renamed businesses with a missing address** (no evidence left to link them).
  - **Near-misses scored just below the threshold**, typically truncated numbers plus appended generic words.
- **France:** the leaderboard gap implies France scores several points below the US/India. Pseudo-label adaptation of the cross-encoder improved France by about 1.5 points (leaderboard 0.966 → 0.969). One experiment wired the French model into a blend slot with a negative weight and hurt France, a lesson recorded in Appendix B.

---

## 6. Conclusion

Strong entity resolution here came from three things working together: blocking designed around measured failure modes (and TF-IDF for the pairs no key can reach), a model that learns ownership competition rather than obeying hard rules, and cross-encoders that read both records together to settle the ambiguous cases. Our main lessons: measure before building (several plausible ideas were rejected in minutes), keep fuzzy evidence in models rather than rules, and watch what validation cannot see (France).

---

## Appendix

### A. Code Artefacts

The complete code is in `code/business_entity_resolution/`: all source in `src/`, a `README.md` with exact run instructions, and a pinned `requirements.txt`.

- **Entry point:** `python src/run_pipeline.py --stage all --raw data/raw --data data/parquet --work work` runs every stage in order and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Any single stage can be rerun with `--stage convert|normalize|block|tfidf|matcher|ce|stack` (`blend` reproduces the earlier T17b configuration).
- **Stage modules:** `t06b_normalize.py` (normalization), `t07_blocking.py` (key blocking), `t15b_tfidf_block.py` (TF-IDF blocking, sharded), `t08_matcher.py` (features, LightGBM, test inference), `t11a_ce_train.py` and `t13_ce_france.py` (cross-encoders), `t19_stack.py` (top-10 scoring shards, second-stage model, final decisions), `t11b_ce_blend.py` (earlier blend configuration).
- **Shared library `src/ber/`:** ids and folds, the exact F0.5 scorer, cleaning, blocking, features, decision rules, and cross-encoder utilities.
- The GPU stages need a CUDA GPU; the final blend stage is exactly reproducible from the saved models and predictions.

### B. Additional Results

**Experiments that did not work (kept for transparency):**

| Idea | Result | Why |
|---|---|---|
| Fuzzy house numbers as a hard rule | −0.05 | More contested records were dropped; fixed by using them as graded features |
| Dropping records claimed by 2+ S1 entities | recall 0.55 → 0.30 | Replaced by competition features + assignment |
| Expected-F0.5 per-entity decision rule | −0.0004 | More false matches on singletons |
| Sibling-based cluster expansion | ceiling +0.007 at 25 extra candidates/S1 | Missed copies are far from every other copy too |
| French model in a negatively weighted blend slot | leaderboard gain 0.003 vs ~0.007 expected | Override must target the positively weighted model |

**Blend on validation half B (T17):** uncertain-pair AUC: LightGBM 0.968, MiniLM-L6 0.979, MiniLM-L12 0.981, MiniLM-L12 v2 0.986, blend 0.990.

**Final stacked model (T19), half B:** F0.5 0.9829 (India 0.9843, US 0.9820), singletons 0.9891. Feature gain share: MiniLM-L12 v2 logit 93%, LightGBM logit 4%, DeBERTa logit 2%. On the test set it predicts matches for 94–95% of S1 entities (3.33–3.47 per entity), consistent across India, US and France.
