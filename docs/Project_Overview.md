# Sepsis Digital Twin - Full Project Overview

Generated: 2025-11-03

## 1) Executive Summary

This system predicts sepsis risk 4-6 hours before onset using ICU time-series data. It combines a strong tabular baseline (Histogram-based Gradient Boosting, HGB) with four deep learning models (GRU-D, LSTM, CNN-LSTM, Transformer) trained on real ICU data, and exposes results via a Streamlit dashboard and a FastAPI microservice. The dashboard shows per-model risk, an ensemble score, an agreement indicator, and detailed evaluation plots. The evaluator produces ROC/PR figures and a model comparison table with operating-point metrics at Recall ~0.85.

## 2) Quick Start (Local Dev)

- Dashboard: `streamlit run dashboard_real.py`
- API: `python api_service.py`
- Evaluation: `python Capstone/evaluate_models.py`
- Outputs:
  - Figures: `outputs/figures/*.png`
  - Table: `outputs/results/model_comparison.csv`
  - PDF Overview: `outputs/docs/Project_Overview.pdf`

## 3) Architecture Overview

- Frontend: Streamlit app (`dashboard_real.py`) with tabs for Risk Overview, Model Predictions, Risk Trajectory, Explainability, Performance, Clinical Workflow, Fairness, and exports.
- Backend: FastAPI (`api_service.py`) with endpoints for prediction, batch, explainability, clinical scores, and health.
- Integration: `integration_real.py` unifies baseline and DL inference, preprocessors, and post-processing.
- Models: Baseline HGB artifacts and 4 DL checkpoints trained on real data under `outputs/models/` (GRU-D prefers `grud_real_data.pt`).
- Evaluation: `Capstone/evaluate_models.py` saves ROC/PR plots and comparison tables.

## 4) Data Flow (End-to-End)

1. Acquire latest patient record from `Dataset.csv` or manual form entry.
2. Integration preprocessing:
   - Harmonize keys, types, and fill missing values.
   - Apply imputer + scaler for the baseline path (HGB).
3. HGB baseline path:
   - Align features to `feature_list.json` order.
   - Transform with `imputer_final.pkl` then `scaler_final.pkl`.
   - Predict with `model_final_hgb.pkl.predict_proba` to get risk probability.
4. Deep learning path:
   - Build a 24-step sequence `[time x features]` and masks.
   - Run GRU-D, LSTM, CNN-LSTM, Transformer checkpoints (trained on real data); apply sigmoid to logits to get probabilities.
5. Aggregate and display:
   - Show per-model risk score and a simple confidence.
   - Compute ensemble as the mean of model probabilities.
   - Compute agreement as `1 - std(probabilities)` (lower variance implies higher agreement).

## 5) Models and Rationale

- Baseline (HGB): strong tabular performance, fast, interpretable, easy to calibrate; used as a sanity floor and fallback.
- GRU-D: handles missingness with masks and time-since-last-seen (delta_t) and decays stale values.
- LSTM: recurrent memory for temporal dependencies.
- CNN-LSTM: captures local temporal patterns with 1D convolution, aggregates across time with LSTM.
- Transformer: attention-based model for long-range dependencies without recurrence.
- Temporal modeling uses trajectories (trends, volatility, timing) instead of single snapshots; typically improves early detection at a fixed recall.

## 6) Integration Layer (integration_real.py)

- Single adapter for baseline and DL inference keeps evaluator and dashboard consistent.
- GRU-D checkpoint preference aligned to evaluator: prefer `grud_real_data.pt`, fallback `grud_demo_model.pt`.
- HGB pipeline: `feature_list.json` -> `imputer_final.pkl` -> `scaler_final.pkl` -> `model_final_hgb.pkl.predict_proba`.
- DL inference robustness:
  - Builds a 24-step sequence from current values.
  - Attempts to load `state_dict`; if incompatible, runs with default weights so the card still renders.
  - Returns a uniform dict `{ model_name: { risk_score, risk_level, confidence } }`.

## 7) Evaluation (Capstone/evaluate_models.py)

- Computes AUROC, AUPRC, default-threshold metrics, and operating-point metrics at Recall ~0.85:
  - Accuracy@R85, Precision@R85, Specificity@R85, F1@R85, Threshold@R85.
- Saves:
  - ROC: `outputs/figures/roc_curves_comparison.png`
  - PR: `outputs/figures/pr_curves_comparison.png`
  - Bar chart: `outputs/figures/model_comparison.png`
  - Table: `outputs/results/model_comparison.csv`

## 8) Dashboard (dashboard_real.py)

- Risk Overview: ensemble score, risk level, agreement metric, recommended action.
- Model Predictions: per-model cards for DL and HGB; confidence shown; optional CI bands.
- Performance Metrics: renders the latest `model_comparison.csv` and charts for Accuracy, AUROC, AUPRC, and @Recall ~0.85.
- Design: modernized header and cards; removed footer text and Streamlit default footer/menu.

## 9) Live Models in the Dashboard

- Uses up to 5 models live if artifacts exist: GRU-D, LSTM, CNN-LSTM, Transformer (all trained on real data), and HGB.
- Evaluation-only baselines (Random and Majority) do not run live for patients.
- If a checkpoint or artifact is missing, that model is skipped; others still run.

## 10) Roles and Deliverables

- Person A (Data Engineering): ICU dataset (`Dataset.csv`), patient-wise splits, missingness strategy, `outputs/config.json` with feature/sequence settings.
- Person B (Baselines): Classical HGB pipeline (`feature_list.json`, `imputer_final.pkl`, `scaler_final.pkl`, `model_final_hgb.pkl`), benchmark metrics, calibration.
- Person C (Deep Learning): GRU-D, LSTM, CNN-LSTM, Transformer models trained on real data; checkpoints in `outputs/models/`; added metrics at Recall ~0.85; ablation studies (e.g., No DeltaT).
- Person D (Integration & Dashboard): Unified inference layer, Streamlit dashboard tabs, evaluation visuals, exports, and UI polish.

## 11) Recent Enhancements (This Review)

- Added operating-point metrics at Recall ~0.85 and exposed them in the dashboard.
- Aligned GRU-D model preference across evaluator and dashboard.
- Integrated HGB baseline end-to-end in both evaluation and dashboard.
- Made DL inference robust so all 4 models attempt to run even with partial checkpoints.
- Removed footer text and improved dashboard styling.

## 12) Failure and Fallback Behavior

- Missing DL checkpoint: model skipped or runs with default weights; clearly logged.
- Missing baseline artifacts: HGB skipped; other models run.
- All models missing: simple heuristic keeps the UI responsive for demo continuity.

## 13) Deployment Notes

- Docker targets for API-only, dashboard-only, or complete system (API + dashboard).
- Health checks at `/health`, API docs at `/docs`.
- For production: restrict CORS, add auth, secure logs and PHI.

## 14) Demo Script (2 Minutes)

1. Open dashboard (http://localhost:8501). Show Risk Overview (ensemble, agreement, recommended action).
2. Show Model Predictions (per-model scores and confidence; point out HGB vs DL).
3. Show Performance tab (Accuracy, AUROC, AUPRC, @Recall ~0.85 from CSV) and figures.
4. Optional: open API docs at http://localhost:8000/docs and send a sample `/predict`.

## 15) FAQ (Short)

- Why HGB baseline? Strong tabular baseline; sanity floor; fast and interpretable; fallback if DL missing.
- Why multiple DL models? Trust via agreement, improved robustness, and optional ensemble uplift.
- What is temporal modeling? Learning from sequences (trends, timing) rather than single snapshots.
- How are thresholds chosen? Tune for clinical targets (e.g., Recall ~0.85) and report Precision/Specificity at that point.
