# Quick Start: Retrain Baseline Models

## Dataset
- **File**: `fully_cleaned_sepsis_data.csv`
- **Location**: Project root directory
- **Size**: ~546,124 rows

## Setup

### Option 1: Use Virtual Environment (Recommended)
```bash
# Activate the virtual environment
source Capstone/venv/bin/activate

# Run training
python train_baseline_models.py
```

### Option 2: Fix pandas Architecture Issue
If you get a pandas architecture error, reinstall pandas for your architecture:
```bash
pip3 install --upgrade --force-reinstall pandas
```

## Training Script
Run the training script:

```bash
python3 train_baseline_models.py
```

## What It Does
1. ✅ Automatically finds `fully_cleaned_sepsis_data.csv`
2. ✅ Computes clinical scores (SIRS, qSOFA, NEWS2, SOFA)
3. ✅ Trains 3 baseline models:
   - Logistic Regression (calibrated)
   - Random Forest (calibrated)
   - XGBoost/HistGradientBoosting (calibrated)
4. ✅ Saves models to `outputs/models/`
5. ✅ Saves metrics to `outputs/models/baseline_models_metrics.json`

## Output Files
- `outputs/models/logistic_regression.pkl`
- `outputs/models/random_forest.pkl`
- `outputs/models/xgboost.pkl`
- `outputs/models/baseline_imputer.pkl`
- `outputs/models/scaler.pkl`
- `outputs/models/baseline_models_metrics.json`

## Compare Results
After training, compare with existing models:

```bash
python3 compare_baseline_models.py
```

