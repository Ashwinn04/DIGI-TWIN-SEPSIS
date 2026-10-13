# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sepsis early-warning system ("Sepsis Digital Twin"): predicts sepsis 4–6 h ahead from ICU time-series. It combines clinical scores (SIRS, qSOFA, NEWS2, SOFA), calibrated baseline ML models (logistic regression, random forest, XGBoost/HistGradientBoosting), and four PyTorch sequence models (GRU-D, LSTM, CNN-LSTM, Transformer). Most files under `docs/` are coursework/presentation material, not developer docs; They now live in `docs/`; `README.md` and `docs/QUICK_START_TRAINING.md` are the useful ones.

## Commands

No test suite or linter is configured. `scripts/test_presentation_setup.py` is only an environment smoke check (imports, files present).

```bash
pip install -r requirements.txt          # torch is installed separately (CPU wheel index) if DL models are wanted
streamlit run dashboard_real.py          # dashboard on :8501
python api_service.py                    # FastAPI (uvicorn) on :8000
python predict_new_data.py --json sample_high_risk_patient.json   # also --csv, --interactive, --data "HR=95,Temp=38.2", --output
python train_baseline_models.py          # baselines -> outputs/models/ (uses fully_cleaned_sepsis_data.csv if present, else Dataset.csv)
TRAIN_ONLY=rf REUSE_PREPROCESSORS=1 python train_baseline_models.py   # retrain one model on the deployed imputer/scaler/selector, merge its metrics
python train_models.py --data_path Dataset.csv --models lstm --epochs 60 --num_seeds 3   # DL models (writes outputs/models/<model>_seed<N>.pt: back up deployed checkpoints first)
python scripts/export_deployment.py      # package the evaluated models -> outputs/deployed/protocol_H6 (calibrators, thresholds)
python scripts/check_deployment.py       # runtime path must reproduce the evaluation's predictions
python predict_new_data.py --json sample_high_risk_patient.json   # serves the evaluated models; SEPSIS_MODEL_SET=legacy for the old checkpoints
python scripts/final_evaluation.py       # THE reported results: all models, one protocol -> outputs/results/final_results.csv (~45 min, cached in outputs/final_run/)
python -m surveillance.run_surveillance --data Dataset.csv --out outputs/surveillance
python scripts/test_presentation_setup.py
```

Docker: multi-stage `Dockerfile` with targets `production` (API), `dashboard`, `complete`, `development`; `docker-compose.yml` wires them.

Datasets (`*.csv`) and `*.pkl` are gitignored, so training data and the baseline pickles (LR, RF, XGBoost, imputer, scaler, selector) are local-only: a fresh clone has none of them and must retrain or copy them across. `Capstone/` is a local virtualenv.

## Architecture

- **`integration_real.py` is the core.** `SepsisPredictionIntegration` (obtained via `get_integration_system()`) loads everything at startup: baseline models, DL checkpoints, imputer/scaler/feature selector, feature names and defaults, and DL ensemble weights from results files. It handles preprocessing (including recomputing clinical scores as features for baselines) and exposes the prediction methods. `TorchModelEnsemble` wraps the multi-seed checkpoints (`<model>_seed42/43/44.pt`) for one architecture and infers input size from the state dict. The dashboard, `predict_new_data.py` and the evaluation scripts all go through this class; changes to feature handling belong here.
- **`dashboard_real.py`** (Streamlit, very large single file) — UI for browsing real data, manual patient entry, CSV upload, and per-model risk plus clinical scores. DL predictions go through `integration_real`.
- **`api_service.py`** (FastAPI) — `/predict`, `/predict/batch`, `/explain`, `/clinical-scores`, `/compare-models`, `/health`. It has its own clinical-score helpers and logs to `outputs/prediction_log.jsonl`. It is a partly separate code path from the integration class, so check whether a fix has to be made in both.
- **`models/`** — PyTorch architectures (`grud.py`, `lstm.py`, `cnn_lstm.py`, `transformer.py`). **`explainability_utils/`** (despite the name) holds the data loading and patient-level splits, training loop (`ModelTrainer`), calibration, metrics, explainability and plotting used by `train_models.py`.
- **`surveillance/`** — separate multi-hospital outbreak-detection layer (out-of-fold risk → simulated hospitals/regions → privacy-preserving aggregation → EWMA/CUSUM). Hospital assignment, dates, outbreaks and viral counts are simulated; see `surveillance/README.md`. `viral_real.py` swaps in a real CDC FluView viral series (`--viral-csv`, data from `scripts/fetch_viral.py` into gitignored `data/viral/`); `sweep.py` runs the sensitivity analysis. `--risk-source <model>` (default `legacy`) swaps the risk scores for leakage-safe, out-of-fold, pre-onset scores from the project's models via `risk_protocol.py`, which shares `explainability_utils/protocol.py` with `scripts/final_evaluation.py`.
- **`scripts/`** — helpers run from the project root (they use cwd-relative `outputs/...` paths); `scripts/docs/` has the markdown→HTML/PDF converters writing to `docs/`.
- **`outputs/`** — checkpoints, scalers/imputers, `thresholds.json`, `config.json`, `feature_list.json`, metrics and results CSVs consumed at runtime. Several `.pt` checkpoints are committed to git. `*_demo_model.pt` are placeholder checkpoints, not trained models (only `scripts/evaluate_accuracy.py` reads them).

## Gotchas

- **Two model sets.** `integration_real.py` serves the `protocol` set by default: the models behind `final_results.csv`, loaded from `outputs/deployed/protocol_H6` via `explainability_utils/deploy.py` (falls back to `legacy` if the bundle is missing). `SEPSIS_MODEL_SET=legacy` serves the older committed checkpoints (trained before the final evaluation, 24-step window, old label definition). Protocol scores are calibrated probabilities near the ~1% base rate; levels use validation operating points (Medium = 80% specificity, High = 95%), not 0.3/0.7. `api_service.py` now calls the same system (it used to return `np.random.uniform` scores; `outputs/prediction_log.jsonl` still contains those fake entries).
- **GRU-D must not depend on its batch.** It used `mean(x, dim=[0, 1])` over the batch for imputation, so a patient's score changed with whoever was scored alongside them. `models/grud.py` now uses 0 (inputs are standardised). Any GRU-D checkpoint trained before that fix (including the legacy ones) was trained with the batch mean.

- **Reported results come from `scripts/final_evaluation.py` only** (`outputs/results/final_results.csv`). Patient split from `outputs/cache/patient_splits_seed42.json`; label = labelled onset within the next H hours, with post-onset hours dropped (89.5% of positive rows in `Dataset.csv` are post-onset continuations and the label never reverts, so a persisted label otherwise inflates AUROC to ~0.95); preprocessing, calibration and threshold fitted on train/validation only. It does not write to `outputs/models/`. Report the `time_only` row alongside the models: hour-in-ICU alone reaches AUROC 0.67 because positives sit late in a stay. `outputs/results/model_comparison.csv` and `test_set_model_comparison.csv` are older, different-protocol numbers that the legacy dashboard path still reads (`evaluation_summary.md` and the 2025 GRU-D run JSONs are in `outputs/results/archive/`).

- **Checkpoint/architecture coupling.** The `*_seed4x.pt` checkpoints were trained with the model classes from commit `5f81942` (kept in `models/`): GRU-D hidden 128, LSTM hidden 128, CNN-LSTM `cnn_filters=[64, 128]`, Transformer d_model 128. `integration_real.py` must build exactly these or the weights load only partially. It loads with `strict=False`, so it now prints a warning when fewer tensors load than the model has; treat that warning as a bug. (History: commit `f03446f` swapped in different GRU-D/LSTM/CNN-LSTM classes, which left those three models on mostly random weights until restored.)
- `forward(features, masks, delta_t=None)`: `delta_t` is optional and defaults to regular hourly steps (0 for the first step, 1 after), so 2-argument callers work. The training loader supplies real time differences.
- **Library pins matter.** The pickles in `outputs/` were created with scikit-learn 1.7.2 (other versions silently break the imputer and the HGB model) and xgboost 2.x (3.x segfaults on `xgboost.pkl`). `requirements.txt` pins both. `integration_real.py` sets `OMP_NUM_THREADS=1` before importing xgboost/torch to avoid a macOS libomp crash/deadlock; keep that before any torch import.
- The deployed baselines use 23 features (Dataset.csv minus 5 pre-engineered columns), defined by `baseline_imputer.pkl`'s `feature_names_in_`. Without `REUSE_PREPROCESSORS=1`, `train_baseline_models.py` re-fits and overwrites the imputer/scaler/selector, which would invalidate the deployed LR/XGBoost. The RF search grid is bounded (depth<=20, leaf>=2) because an unbounded search produced a 1.3 GB pickle.
- Baseline scores are calibrated row-level probabilities (low base rate, e.g. LR tops out at 0.25), while the deep-learning models were trained with class weighting and score higher. Do not compare the two scales directly.
- `train_models.py` writes to the same `outputs/models/` paths the dashboard loads from, and `outputs/cache/patient_splits_seed42.json` fixes the patient split.
- Model input size and sequence length must match the checkpoint and `outputs/config.json` (44 features, 24 steps). Retraining with different features means regenerating the imputer, scaler and feature list.
- Several scripts in `scripts/` (`calculate_baseline_accuracy.py`, `compare_baseline_models.py`, `evaluate_accuracy.py`) do not run without the training dataset.
