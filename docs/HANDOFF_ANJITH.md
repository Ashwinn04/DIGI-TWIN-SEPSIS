# Handoff for Anjith: what changed in the model and evaluation code (branch `ash2`)

Written for Anjith (Person C, deep learning). Everything below is in `ash2`; none of it is on `main` yet.
Please read it before merging anything or pushing from a branch based on `f03446f`.

## 1. Why this matters
- The 0.92-0.96 AUROC figures in the paper/slides cannot be traced to any run in the repository (all 59
  commits searched). Under a leakage-safe protocol the honest numbers are about 0.74-0.80 (below).
- 89.5% of positive rows in `Dataset.csv` are *post-onset continuations* and the label never reverts. A
  "model" that copies the previous hour's label scores AUROC 0.948 at row level. Evaluating on those rows
  inflates every number.

## 2. Results to use (single source of truth): `outputs/results/final_results.csv`
Produced by `scripts/final_evaluation.py` (one patient split, label = labelled onset within the next H hours
with post-onset hours dropped, preprocessing fitted on train only, threshold/calibration on validation,
3 seeds, 1000 patient-level bootstrap resamples, H = 3/6/12). H=6, test, AUROC [95% CI]:

| Model | AUROC | AUPRC |
|---|---|---|
| Time-in-ICU only | 0.673 [0.62, 0.73] | 0.067 |
| GRU-D | 0.796 [0.76, 0.83] | 0.057 |
| CNN-LSTM | 0.793 [0.76, 0.82] | 0.053 |
| XGBoost | 0.787 [0.75, 0.82] | 0.058 |
| Transformer | 0.784 [0.75, 0.82] | 0.061 |
| LSTM | 0.777 [0.74, 0.81] | 0.051 |
| Random forest | 0.775 [0.74, 0.81] | 0.057 |

Models are statistically comparable; hour-in-ICU alone has the highest AUPRC. Do not write "outperforms".

## 3. What changed in your code
1. **`models/` restored to the `5f81942` classes.** Commit `f03446f` ("DL models") swapped in different
   GRU-D/LSTM/CNN-LSTM architectures but kept checkpoints trained with the older ones; the system was
   loading them with `strict=False` and running on mostly random weights (GRU-D 2/11 tensors, LSTM 2/18).
   All committed `*_seed4x.pt` load strictly into the restored classes. If `f03446f`'s architectures were
   intended, their checkpoints are not in the repo; tell us.
2. **GRU-D no longer depends on its batch** (`models/grud.py`). The imputation mean was
   `mean(x, dim=[0, 1])` over the batch, so a patient's score changed with the other patients scored with
   them (up to 0.16). It is now 0 (inputs are standardised). GRU-D was retrained; accuracy unchanged.
3. **`train_models.py`** is a single script again (two had been concatenated; three helper functions it
   imports were missing) and the data loader/trainer are the `5f81942` versions with optional `delta_t`.
4. **`explainability_utils/protocol.py`** holds the shared labels/preprocessing/windows/models used by both
   `scripts/final_evaluation.py` and `surveillance/risk_protocol.py`.
5. **Your "improved" models** (`bbe1ee5`, reverted by `fb76f20`) exist only in git history. They load
   strictly, but they were trained on a different task (48-step window, 6 h horizon, old label) and have
   not been evaluated under the new protocol. Decide whether to re-evaluate or archive them.

## 4. Serving
`SEPSIS_MODEL_SET` selects the models: `protocol` (default) serves the bundle in
`outputs/deployed/protocol_H6` (build with `scripts/export_deployment.py`, verify with
`scripts/check_deployment.py`); `legacy` serves the older checkpoints. The REST API used to return
`np.random.uniform` scores; it now calls the same system. Scores are calibrated probabilities near the 1%
base rate; Medium/High are the 80%/95% specificity operating points, not 0.3/0.7.

## 5. What we need from you
- [ ] Rerun `python scripts/final_evaluation.py` on your machine from a fresh clone of `ash2` and compare
      with `final_results.csv` (we reproduced AUROC/AUPRC to 4 decimals here). This is the plan's "second
      person reruns it" check.
- [ ] Tell us where 0.92-0.96 came from (notebook/script and date), or confirm it is unverified.
- [ ] Rebase your work on `ash2`; do not restore `f03446f`'s `models/`.
- [ ] Confirm which architectures/checkpoints are final.
- [ ] Review the chapter's model section against `final_results.csv` (no number typed from memory).

## 6. Known limits (say them in the paper)
One dataset, one split; effective positives come from ~100 test patients; calibration skill is small
(Brier 2-3% better than predicting the prevalence); precision at the F1 point is ~9-11%; lead time before
onset was not measured; the dataset's labelled onset is already ~6 h before clinical onset.

## 7. Anjith's response (2026-10-13, branch `ash2`)
- [x] **Rerun from a fresh clone of `ash2`** (`--horizons 3 6 12 --sanity`, macOS arm64, Python 3.12, torch 2.14.1;
      environment in `requirements.lock.txt`). Results: `outputs/results/reproduction/anjith_*.csv`.
  - Same data, split and labels: `n_test`, `n_test_pos`, `n_test_patients` and all 12 sanity-check AUROCs identical.
  - **Bit-identical (AUROC diff 0):** time-only, LR, decision tree, KNN, RF, XGBoost, HistGB, GRU-D. LSTM within 2e-4.
  - **Differ:** Transformer (per-seed up to 0.047 at H=3; H=6 mean 0.777 vs 0.772), CNN-LSTM (up to 0.004),
    ANN (H=12 mean 0.754 vs 0.772, the only seed mean outside 2 std). Retrained Transformer/CNN-LSTM weights differ
    from the deployed ones, so their training is not bit-reproducible across machines.
  - H=6 seed means agree within noise for every model. Conclusion for the chapter is unchanged: models are
    statistically comparable at ~0.74-0.79. Report mean +/- std over seeds, not single-seed or ensemble rows.
  - `scripts/check_deployment.py --workdir <my rerun>`: GRU-D, LR, XGBoost OK; LSTM/CNN-LSTM/Transformer MISMATCH only
    because my retrained checkpoints differ from the bundle (expected). Needs your cached `outputs/final_run/preds`
    to check the bundle itself; it cannot run from a fresh clone.
- [x] **0.92-0.96:** placeholder table never replaced; see `docs/RESULTS_PROVENANCE.md`.
- [x] **Architectures:** the restored `5f81942` classes in `models/` are final. My `bbe1ee5` "improved" models are
      archived, not re-evaluated: tag `archive/anjith-improved-models`. Do not report them.
- [x] **Rebased on `ash2`.**
- [ ] **Chapter review against `final_results.csv`:** waiting for the chapter draft.

Also added: committed patient split (`outputs/cache/patient_splits_seed42.json`), `requirements.lock.txt`,
`scripts/test_setup.py`, README "Reproduce the paper results", stale results moved to `outputs/results/archive/`.

## 8. Ashwin's response to section 7 (2026-10-13)
- **Reproduction accepted.** Cross-machine comparison (`reproduction/anjith_final_results.csv` vs `final_results.csv`,
  108 rows matched): 8 of 12 models are bit-identical (time-only, LR, DT, KNN, RF, XGBoost, HistGB and **GRU-D**,
  which confirms the batch-independence fix removed its nondeterminism); LSTM within 2e-4; CNN-LSTM per-seed
  up to 0.004; Transformer per-seed up to 0.047 (seed ensemble 0.018); ANN per-seed up to 0.028. Ranking of the
  top models is unchanged. For the chapter: report **mean +/- std over seeds** (`final_results_summary.csv`) and say
  Transformer/CNN-LSTM/ANN training is not bit-reproducible across machines.
- **`scripts/check_deployment.py` now runs from a fresh clone** (needs `Dataset.csv`, no cache, no training): it
  recomputes the evaluation path from the bundle's own checkpoints and compares the full-test AUROC/AUPRC with the
  values recorded in `calibration.json`. All 7 bundle models match to 4 decimals.
- **Extra finding supporting your "no missing values" check:** the 44 model columns have 0 missing cells and the
  slow labs are carried forward (share of hours equal to the previous hour: lactate 92.6%, creatinine 85.8%,
  platelets 85.0%, glucose 76.3%; vitals: HR 11.6%, Temp 48.7%). So `Dataset.csv` was forward-filled (or similar)
  before it reached us, the observed-mask is all ones apart from window padding, and **GRU-D's missing-data
  handling cannot be demonstrated on this file**. The chapter should not claim a missingness advantage. Whoever
  prepared the data (Person A) should state the imputation method; if it used cross-patient statistics computed
  before the split, that is a (small) leakage path we cannot rule out from here.
- **Still open:** chapter review against the tables (waiting for the draft); suppression normalisation and a
  geographic-localisation metric in `surveillance/`.

## 9. Answers to Anjith's "Need from you" (Ashwin, 2026-10-13)
1. **[9]/[10] references.** Checked against the tracked manuscript's own sentences. [10] Tang, Zhang & Li (BMC Med Res
   Methodol 2024, transformer for early sepsis prediction) fits the sentence it supports ("Transformer-based models ...
   self-attention ... [6], [7], [10]"). [9] Fabrizzio et al. (Rev Bras Enferm 2023, web app predicting ICU
   hospitalisation for COVID-19) fits only loosely: it is cited for "clinician-facing ... alerts ... improving clinical
   decision-making and patient outcomes [9],[11],[13]" and for the Digital Twin dashboard sentence [3],[9],[11],[14].
   It is a COVID web app, not sepsis or a digital twin, and a patient-outcomes claim needs outcome evidence. Soften
   those two sentences or cite [9] only for "a web application for ICU risk prediction". I could not open either paper
   from here: confirm both on the publisher/DOI before accepting.
2. **Figures 5 and 6 were dashboard placeholders, not results.** Both are screenshots (dated 31 Oct - 1 Nov 2025) of
   `generate_risk_trajectory` (`0.3 + sin(0.1 i)*0.1 + Gaussian noise`, no patient data, no model) and
   `generate_feature_importance` (`|value - reference| / reference` for 8 fixed features; not SHAP). The text and
   captions ("the model's predicted probability of sepsis at that hour", "SHAP summary ... top drivers across models")
   are therefore false and must be replaced. Real replacements: `scripts/make_figures_2_3.py` (manuscript Figures 2 and 3; renamed
   from 5 and 6) -> `outputs/figures/figure2_risk_trajectory.png` (two held-out patients chosen by a fixed rule; hour-by-hour calibrated
   digital-twin ensemble) and `figure3_shap_xgboost.png` (exact TreeSHAP of the XGBoost model, test pre-onset hours),
   with the numbers in `outputs/results/figure2_trajectory.csv` and `figure3_shap_mean_abs.csv`.
   **The real SHAP result contradicts the draft text:** top features are ICULOS (hour in ICU, 0.74), HospAdmTime (0.32),
   Unit1, Age, MAP_rolling_mean_6hr, then HR, Resp; lactate and WBC are not in the top ten. Time in ICU dominates,
   consistent with the time-only baseline (AUROC 0.67). Say "SHAP for the XGBoost model", not "across models".
3. **How `Dataset.csv` was imputed** (answers your open question). Every missing hour holds the **dataset-wide median**
   of its column, with no carry-forward and no backfill: real values are isolated single hours (platelets: mean run of
   identical non-median values 1.00 h; creatinine repeats with probability 0.004), and 87-100% of patients show one
   identical value (= the global median) in their first and last rows (TroponinI 4.75 for 100%). So there is **no
   future-value leakage** from backfill. Two consequences: (a) the median was computed on the whole dataset before
   the split (one scalar per column, negligible but a pre-split statistic); (b) "value != median" is a *measurement
   indicator*. Measurement indicators alone give XGBoost AUROC 0.58 (6-h measurement counts 0.60) against 0.78 with
   values and 0.67 for time-only, and pre-onset hours have more labs measured (1.10 vs 0.86 per hour): informative
   missingness exists but is a small part of the skill. GRU-D's missing-data handling is not exercised on this file.
4. **`check_deployment`.** Runs from a fresh clone now (`git pull`, then `python scripts/check_deployment.py`); no cache or
   RF pickle needed (it skips baselines whose pickle is absent).
5. **RF pickle / pip freeze.** The RF pickle (20 MB) is local-only (`*.pkl` gitignored) and is not needed for the check
   above. Environment on this machine: Python 3.13, scikit-learn 1.7.2, xgboost 2.1.4, torch 2.14.1 (see `requirements.txt`).
6. **Dashboard.** The remaining placeholders are gone too: the risk trajectory is now hour-by-hour model output (empty for
   a single manual entry), alert persistence and chart thresholds use the score's own operating points, the ensemble
   level comes from the system, and the "Top Contributing Features" chart is labelled as deviation from reference range.
   Fixed a regression: with calibrated scores near 0.03, the old fixed 0.3/0.7 cut-offs would have shown "Low" for everyone.
