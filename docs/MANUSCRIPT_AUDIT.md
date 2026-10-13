# Audit of `514_Manuscript_revised_tracked.docx` against the repository (2026-10-13)

Paragraph numbers are positions in the `.docx` body (insertions included, deletions excluded).
Note: `514_Manuscript_revised_tracked.pdf` shows the **original** text (placeholder Table 2, old [9]/[10], old
author order, no surveillance section); it is not a render of the tracked `.docx`. Review the `.docx`.

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
   Fig. 6 is a deviation-from-reference heuristic, not SHAP. Replace with `outputs/figures/figure5_risk_trajectory.png`
   and `figure6_shap_xgboost.png` (`scripts/make_figures_5_6.py`) and rewrite 293-303. Suggested text:
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
