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
python train_baseline_models.py          # baselines -> outputs/models/ (needs fully_cleaned_sepsis_data.csv in root)
python train_models.py --models grud lstm --epochs 60 --num_seeds 3   # DL models; see args at bottom of file
python -m surveillance.run_surveillance --data Dataset.csv --out outputs/surveillance
python scripts/test_presentation_setup.py
```

Docker: multi-stage `Dockerfile` with targets `production` (API), `dashboard`, `complete`, `development`; `docker-compose.yml` wires them.

Datasets (`*.csv`) and `*.pkl` are gitignored, so training data is not in the repo. `Capstone/` is a local virtualenv.

## Architecture

- **`integration_real.py` is the core.** `SepsisPredictionIntegration` (obtained via `get_integration_system()`) loads everything at startup: baseline models, DL checkpoints, imputer/scaler/feature selector, feature names and defaults, and DL ensemble weights from results files. It handles preprocessing (including recomputing clinical scores as features for baselines) and exposes the prediction methods. `TorchModelEnsemble` wraps the multi-seed checkpoints (`<model>_seed42/43/44.pt`) for one architecture and infers input size from the state dict. The dashboard, `predict_new_data.py` and the evaluation scripts all go through this class; changes to feature handling belong here.
- **`dashboard_real.py`** (Streamlit, very large single file) — UI for browsing real data, manual patient entry, CSV upload, and per-model risk plus clinical scores. DL predictions go through `integration_real`.
- **`api_service.py`** (FastAPI) — `/predict`, `/predict/batch`, `/explain`, `/clinical-scores`, `/compare-models`, `/health`. It has its own clinical-score helpers and logs to `outputs/prediction_log.jsonl`. It is a partly separate code path from the integration class, so check whether a fix has to be made in both.
- **`models/`** — PyTorch architectures (`grud.py`, `lstm.py`, `cnn_lstm.py`, `transformer.py`). **`explainability_utils/`** (despite the name) holds the data loading and patient-level splits, training loop (`ModelTrainer`), calibration, metrics, explainability and plotting used by `train_models.py`.
- **`surveillance/`** — separate multi-hospital outbreak-detection layer (out-of-fold risk → simulated hospitals/regions → privacy-preserving aggregation → EWMA/CUSUM). Hospital assignment, dates, outbreaks and viral counts are simulated; see `surveillance/README.md`. `viral_real.py` swaps in a real CDC FluView viral series (`--viral-csv`, data from `scripts/fetch_viral.py` into gitignored `data/viral/`); `sweep.py` runs the sensitivity analysis.
- **`scripts/`** — helpers run from the project root (they use cwd-relative `outputs/...` paths); `scripts/docs/` has the markdown→HTML/PDF converters writing to `docs/`.
- **`outputs/`** — checkpoints, scalers/imputers, `thresholds.json`, `config.json`, `feature_list.json`, metrics and results CSVs consumed at runtime. Several `.pt` checkpoints are committed to git. `*_demo_model.pt` are placeholder checkpoints, not trained models (only `scripts/evaluate_accuracy.py` reads them).

## Gotchas

- `train_models.py` defines `main()` twice, and `train_model` sits between them. The second definition (defaults: 50 epochs, batch 32, sequence length 24, horizon 4) overrides the first (60 epochs, batch 128, sequence length 48, horizon 6, multi-seed). The arguments actually in effect are therefore the second set.
- Model input size and sequence length must match the checkpoint and the feature list in `outputs/`. Retraining with different features means regenerating the imputer, scaler and feature list, or `integration_real` will fail to load or silently use defaults.
- Several scripts in `scripts/` (`calculate_baseline_accuracy.py`, `compare_baseline_models.py`, `evaluate_accuracy.py`) do not run without the training dataset.
