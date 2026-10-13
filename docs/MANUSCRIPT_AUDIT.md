# Audit of `514_Manuscript_revised_tracked.docx` against the repository (2026-10-13)

Paragraph numbers are positions in the `.docx` body (insertions included, deletions excluded).
Note: `514_Manuscript_revised_tracked.pdf` shows the **original** text (placeholder Table 2, old [9]/[10], old
author order, no surveillance section); it is not a render of the tracked `.docx`. Review the `.docx`.

> **File names (renamed 2026-10-13 to match the manuscript's final figure numbers):** the new Figures 2 and 3 are
> `outputs/figures/figure2_risk_trajectory.png` and `figure3_shap_xgboost.png` (were figure5_/figure6_), the data-flow
> diagram is `figure4_dataflow.png` (was figure7_). "Figure 5/6" in the text below refers to the *old* placeholder
> numbering.

## Verified correct
- **Table 2** equals `outputs/results/final_results_summary.csv` (mean +/- std over seeds, H = 6 h, all 12 rows).
- **Table 3** equals `outputs/surveillance/leakage_safe/grud_real_viral/surveillance_results.json` (retrained,
  batch-independent GRU-D; real viral series; 30 replicates).
- Methods text matches the code: 70/15/15 patient split, onset-relative label with post-onset hours dropped,
  24-hour zero-padded windows with an observed mask, Platt calibration and F1 threshold on validation, AdamW with a
  cosine schedule, three seeds, patient-level bootstrap.
- Placeholder notes, the fake Figures 2/3 (hard-coded AUROC/AUPRC bars) and the "confidence values" paragraph are gone.
- Headline "0.696-0.787 for eleven models" matches the table (time-only 0.673 is the reference).

## Still wrong (fix before submission)
1. **Figures 5 and 6 and paragraphs 290-303 are the old placeholders.** Fig. 5 is `0.3 + sin + Gaussian noise`;
   Fig. 6 is a deviation-from-reference heuristic, not SHAP. Replace with `outputs/figures/figure2_risk_trajectory.png`
   and `figure3_shap_xgboost.png` (`scripts/make_figures_2_3.py`) and rewrite 293-303. Suggested text:
   - Fig. 5 caption: "Hour-by-hour risk from the Digital Twin (average of four deep sequence models, three seeds each)
     for two held-out test patients: a septic patient over the 24 hours before labelled onset (left) and a non-septic
     patient over the last 24 hours of the stay (right). The patients were chosen by a fixed rule (septic patient with
     onset closest to the median onset; non-septic patient of most similar length), not for appearance. Dashed lines
     are the Medium and High operating points (80% and 95% specificity on validation)."
   - Fig. 5 text: "In the septic patient the calibrated probability of onset within 6 h rose from 0.028 to 0.069 over
     the 24 hours before labelled onset and crossed the High threshold (0.032) about 20 hours before onset; the
     non-septic patient peaked at 0.030, below it. This is one illustrative case, not an estimate of average lead time,
     which was not measured." Delete "The model identifies these deviations earlier than conventional scoring systems".
   - Fig. 6 caption: "Mean absolute TreeSHAP value (log-odds) of the ten most important features of the XGBoost model
     on held-out pre-onset hours."
   - Fig. 6 text: "Hour in ICU dominates (mean |SHAP| 0.74), followed by hours since hospital admission (0.32), unit
     (0.21), age (0.21) and the 6-hour mean arterial pressure (0.18); heart rate and respiratory rate follow (0.15).
     Lactate and white-cell count are not among the ten largest. The dominance of time in ICU agrees with the
     length-of-stay-only reference (AUROC 0.673) and shows that part of the discrimination reflects how long the
     patient has been in the unit." Delete "Heart rate, lactate, MAP trends, and WBC count were among the top drivers
     across models" (it explains one model, and lactate/WBC are not top).
2. **SHAP claims (143, 151, 155, 161) describe a feature that is not in the running system.** The dashboard shows
   reference-range deviations, the API's `/explain` returns null SHAP/LIME/gradients, and SHAP exists only as an
   offline analysis of XGBoost. Reword to "SHAP explanations are available for tree models; the dashboard
   currently reports deviations from reference ranges".
3. **Real-time / continuous-ingestion claims (14, 23, 31, 143, 153, 159).** The system scores hourly windows from
   a file; no streaming deployment or latency was evaluated. Use "hourly risk scoring" and add that it was evaluated
   offline.
4. **Imputation (128) and masks (126, 139).** `Dataset.csv` arrived already imputed: every missing hour is the
   dataset-wide median (no carry-forward), the median was computed before the split, and the observed mask is all
   ones apart from window padding. Say so; our train-only imputer was effectively a no-op. State that GRU-D's
   missing-data handling was not exercised and that "value != median" is a measurement indicator (indicators alone:
   AUROC 0.58).
5. **Outcome claim (23):** "improving clinical decision-making and patient outcomes [9], [11], [13]" has no support
   in this work; remove or cite as motivation only. [9] (Fabrizzio 2023) is a COVID-19 ICU web app, not sepsis or a
   digital twin: use it only for "a web application for ICU risk prediction". Verify [9]/[10] on their DOIs.
6. **Table 2 labelling.** Point estimates are the mean over seeds, but the CI columns are bootstrap intervals of the
   seed-ensemble predictions (GRU-D ensemble 0.796 [0.761, 0.829], table shows 0.787 +/- 0.009 and 0.761-0.829).
   Say so in the caption.
7. **Reference [4] (MIMIC-III):** keep only if cited for context; the data used are PhysioNet-2019-style
   (14,057 patients), not MIMIC.

## Not claimed anywhere (keep it that way)
Lead time before onset, calibration quality beyond a 2-3% Brier skill, external validation, clinical benefit.

---
# Round 2: `514_Manuscript_revised_tracked-2.pdf` (insertions rendered), checked 2026-10-13

## Numeric claims checked against the repository (all correct unless noted)
- Abstract/Results: eleven models, AUROC 0.696-0.787; Table 2 = `final_results_summary.csv`; 2,092 test patients,
  80,098 hours, 860 positive (1.1%); 2.2% of records / 8.8% of patients positive; 89.5% post-onset; 0.948.
- H = 3/6/12 means: XGBoost 0.780/0.784/0.799 and GRU-D 0.793/0.787/0.793 exactly as in the text.
- Shuffled labels 0.40-0.41, within-6-h-bin shuffle 0.69; AUPRC 4-6x chance (0.042-0.067 vs 0.0107); precision
  at the F1 threshold 8.6-10.8%; Brier skill 2-3%.
- Reproduction sentence: at H = 6 the Transformer, CNN-LSTM and MLP seed means differ between machines by less
  than one seed SD (0.0046 vs 0.0055; 0.0012 vs 0.0105; 0.0003 vs 0.0241). (At H = 12 the MLP differs by 0.018, more than
  its SD; the text does not claim otherwise.)
- Surveillance setup: 13,920 patients, patient-level AUROC 0.743, weeks from epiweek 202436, accessed 9 Oct 2026,
  onset days 63-81, 12 sites / 4 regions / 140 days / 21-day surge / 60%; Table 3 = current retrained-GRU-D run.
- Sensitivity paragraph (all EWMA, combined signal): eps 0.1 -> 60% detected, 17.5 d; eps >= 0.5 -> 100%; delay 8 d
  (eps 0.5) -> 3 d (none); k = 10 false alarms 0.61 -> 2.69; intensity 0.2 -> 9 d, 0.8 -> 4 d; viral alone 33% at
  1.5-fold, 100% at 5-fold; viral with 1 site/region 67%, 5 sites 100%; control 0.54 per 100 region-days.
- References [1]-[8], [11]-[30] have the right authors/venues/years/volumes as far as I can tell from memory;
  not checked online. [9], [10] still need a check on their DOIs.

## Must fix
1. **Figures 5 and 6 (page 7) are still the placeholder images and captions.** Replace with
   `outputs/figures/figure2_risk_trajectory.png` / `figure3_shap_xgboost.png` and the text in Round 1, item 1.
   Figure numbers now jump 1 -> 5 (old 2-4 were removed): renumber 5-8 to 2-5 and fix the in-text references.
2. **Real-time / SHAP / outcome claims remain**: abstract ("real-time visualization"), Introduction ("real-time
   visualization ... improving clinical decision-making and patient outcomes [9], [11], [13]"), Digital Twin
   Framework ("continuously ingests ... in real time ... generates SHAP-based explanations"), "End-To-End System
   Architecture" and "Architectural components" (streaming collector, "SHAP values are computed for the current
   prediction"), Table 1 "Present Work: Real-time EHR data". What exists: offline hourly scoring from a file, a
   dashboard that shows reference-range deviations, and an offline TreeSHAP analysis of XGBoost.
3. **Dataset section, imputation and mask paragraphs are inaccurate.** `Dataset.csv` was median-filled before we
   received it (dataset-wide median per column, no carry-forward); our train-only imputer is a no-op and the observed
   mask is all ones except window padding, so "missingness is visible to the model" and "GRU-D explicitly models
   missingness" are not exercised here. Say so, with the measurement-indicator result (AUROC 0.58).
4. **"formal differential-privacy protection" (Literature Review) needs the accounting.** Laplace noise with scale
   1/eps is added to each of three released counts; one patient contributes to at most two of them per day
   (admissions, alerts), so the per-patient budget is 2*eps (3*eps if viral positives were patient-derived), and this
   assumes each patient appears on one day at one site, as in the simulation. State that, or write
   "noise calibrated to eps per released count" and drop "formal".
5. **"about 100" positive test patients (Limitations) is 155** (860 positive hours from 155 of 2,092 patients).
6. **State that the sensitivity numbers are EWMA** (CUSUM differs slightly, e.g. control 0.77 false alarms), and
   add the chance floor: in the no-outbreak control an alarm falls in the placebo window in 10% (EWMA) to 17%
   (CUSUM) of replicates, so weak settings (viral 1.5-fold, eps 0.1) sit near chance detection.

## Smaller points
- "At 3 and 12 hours the same models lead": at 3 h CNN-LSTM is first (0.796), GRU-D second (0.793), XGBoost fourth.
  Write "the same group of models leads".
- "SHAP provides feature-level explanations" (Implementation): the values in Fig. 6 are XGBoost's built-in exact
  TreeSHAP (`pred_contribs`), not the `shap` package. [4] (MIMIC-III) is cited for general ML capability while MIMIC
  was not used; consider citing [5] there.
- The single-hospital result depends on how thin each site is (with one site per region it detects 100% in 3 days,
  as well as pooling). The Limitations already say each hospital has few patients; add that the 3-4 day gain is
  specific to 12 sites sharing the pool.
- "To our knowledge, no prior work pools alerts from bedside sepsis-risk models across hospitals ...": a novelty
  claim that nobody here has searched for; verify or soften.
