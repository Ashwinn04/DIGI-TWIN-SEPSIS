# n Person C – Deep Learning Improvement

# Roadmap

```
This roadmap outlines targeted improvements for Person C’s role in the ClinTwin – Clinical Digital
Twin for Early Sepsis Prediction project. It identifies upgrades that will increase AUROC/AUPRC
performance, improve interpretability, and enhance research-grade quality for publication.
```
## A. Data & Input Representation

Improvement Description Impact
Temporal gap encoding Include ∆t (time since last measurement) per feature.Improves time-aware modeling (GRU-D/Transformer).
Mask vectors & missingness embeddingsFeed binary masks and learn missingness embeddings.Prevents misinterpretation of missing values.
Physiological feature engineeringAdd deltas, rolling means, shock index, SOFA sub-scores.Captures physiological trends, boosts interpretability.
Dynamic horizon labeling Predict onset at 3h, 6h, 12h horizons. Improves flexibility and timeliness evaluation.

## B. Model Architecture Enhancements

Upgrade Purpose Expected Gain
Attention-based LSTM/GRU-DAdd temporal attention for vital signs. +0.05 AUROC
Time-aware Transformer Use Informer/Transformer-TS architecture. +0.07 AUROC
Multi-task Learning Predict both Sepsis risk & Time-to-onset. Richer temporal encoding
Stacked Ensemble Combine XGBoost + DL predictions. Improved robustness

## C. Training Strategy Improvements

- Implement Focal Loss for class imbalance (2% sepsis prevalence).
- Perform threshold tuning using Youden J or F1-max.
- Add early stopping, learning rate scheduler, and mixed precision training.
- Use 5-fold patient-split cross-validation for robustness.

## D. Evaluation Enhancements

- Add Calibration (Brier Score, Reliability Diagram).
- Compute Timeliness metric (hours before onset detected).
- Report Confidence Intervals via bootstrapping.
- Validate externally on eICU or PhysioNet datasets.

## E. Explainability Hooks (for Person D Integration)

Feature Tool Benefit
Feature Importance Export Captum / SHAP Helps visualize top contributing vitals/labs.
Attention Heatmaps Integrated in Transformer Visual explanation for clinicians.


Attribution Storage Save .npz per inference Reusability for dashboard visualization.

## F. Research & Reporting Improvements

- Run 3 training seeds and report mean ± std for reproducibility.
- Add ablation studies (with/without masks, ∆t, labs).
- Compare with literature (Nemati et al. 2018, Moor et al. 2021).
- Provide model cards and config files for reproducibility.

## n Expected Outcome After Improvements

Metric Current Target
AUROC 0.56 (Transformer) ≥ 0.
AUPRC 0.02 ≥ 0.10 – 0.
Timeliness (lead time) Not measured 4–6 hours before onset
Calibration (Brier Score) – ≤ 0.
Explainability Integration Basic Captum & SHAP-enabled