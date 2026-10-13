# Provenance of the reported AUROC numbers (revision plan, Step 2)

Checked 2026-10-13 by Anjith. Question: where did Table I of the paper/slides (AUROC 0.92-0.96) come from?

## Verdict: unverified, placeholder values. Do not use any of them.

The results section of `Research Paper(CAPSTONE).docx` (Dec 16, 2025) introduces the table with:

> "Below are representative results (placeholders — replace with your experiment's actual numbers).
> Figures are included as placeholders; replace the image files in Overleaf with your actual plots"

The table was never replaced. Its values also rise strictly with model complexity in every column
(Accuracy 0.842 -> 0.939, Recall 0.781 -> 0.927), which is what a hand-written template looks like.

| Reported number (paper / slides) | Source found | Status |
|---|---|---|
| Transformer AUROC 0.961 | placeholder table only | unverified |
| CNN-LSTM AUROC 0.958 | placeholder table only | unverified |
| LSTM AUROC 0.953 | placeholder table only | unverified |
| GRU-D AUROC 0.944 | placeholder table only | unverified |
| XGBoost AUROC 0.931 | placeholder table only | unverified |
| Random Forest AUROC 0.924 | placeholder table only | unverified |
| Logistic Regression AUROC 0.861, Accuracy 0.842 | placeholder table only | unverified |
| All Accuracy / Precision / Recall values in Table I | placeholder table only | unverified |
| "LSTM and RF AUROC 0.95 and 0.92" (`Sepsis_Research_Paper.docx`, `Sepsis_AIP_FULL_CONTENT.docx`) | no run | unverified |
| "GRU-D AUROC approx. 0.91, Transformer/CNN-LSTM approx. 0.90" (`SEPSIS DETECTION REPORT.docx`, Dec 5) | no run; the same report later states GRU-D AUROC 0.513 | unverified |

## Where we looked
- All commits on every branch of this repo (also searched by Ashwin, see `docs/HANDOFF_ANJITH.md`):
  none of 0.961 / 0.958 / 0.953 / 0.944 / 0.931 / 0.924 / 0.842 is printed by any script or notebook.
- `notebooks/Capstone.ipynb` outputs: AUROC 0.61-0.79 (highest 0.7914).
- `outputs/results/test_set_model_comparison.csv`, `evaluation_summary.md`: deep models 0.48-0.54
  (23-patient run, now in `outputs/results/archive/`).
- Paper drafts in Anjith's Downloads (Nov 24 - Dec 16, 2025): `Research Paper(CAPSTONE).docx`,
  `Sepsis_Research_Paper.docx`, `SEPSIS DETECTION REPORT.docx`, `Sepsis_AIP_*.docx`.
- Not found on Anjith's machine: `Capstone (1).ipynb`, `Capstone (4).ipynb` (Coursework folder).
  Ashwin / other co-authors: please check yours and any Colab revision history.

## What replaces them
`outputs/results/final_results.csv` from `scripts/final_evaluation.py` (leakage-safe protocol).
H = 6 h, mean of 3 seeds, test AUROC 0.74-0.79 for all models; hour-in-ICU alone 0.67.

## Other claims in the draft to fix (Step 7)
- "ICU-style datasets that resemble ... PhysioNet ... and MIMIC-derived cohorts": only `Dataset.csv`
  (PhysioNet 2019-style, 14,057 patients) was used. Do not cite MIMIC as a data source.
- "Transformer and CNN-LSTM achieved the highest ... AUROC": not supported; models are within each other's CIs.
