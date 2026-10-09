# Sepsis Digital Twin - AI-Powered Early Detection System

A comprehensive sepsis prediction system that combines clinical scoring, traditional machine learning, and deep learning models for early sepsis detection 4-6 hours before onset using ICU time-series data.

## Project Overview

The Sepsis Digital Twin system integrates multiple prediction approaches:

### Deep Learning Models
- **GRU-D**: Handles missing data with time-decay mechanism
- **LSTM**: Bidirectional LSTM for sequential modeling
- **CNN-LSTM**: Hybrid architecture for local temporal patterns
- **Transformer**: Attention-based modeling for long-range dependencies

### Traditional Machine Learning Models
- **Logistic Regression**: Baseline linear classifier
- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting for sepsis risk classification

### Clinical Scoring Systems
- **SIRS**: Systemic Inflammatory Response Syndrome criteria
- **qSOFA**: Quick Sequential Organ Failure Assessment
- **NEWS2**: National Early Warning Score 2
- **SOFA**: Sequential Organ Failure Assessment

### Interactive Dashboard
- **Real-Time Patient Monitoring**: Analyze ICU patients with live data
- **Manual Patient Entry**: Add and analyze custom patient profiles
- **Comprehensive Risk Analysis**: Clinical scores + ML/DL model predictions
- **Explainable AI**: Feature importance and model interpretability

## Quick Start

### Running the Interactive Dashboard
```bash
# (Recommended) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate    # Windows

# Option A: With Deep Learning support (if you have DL checkpoints)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt

# Option B: Baseline-only (skip PyTorch)
# pip install -r requirements.txt

# Start the dashboard
streamlit run dashboard_real.py
```

Then open your browser to: **http://localhost:8501**

### Features Available
1. **Browse Real Data**: Analyze ICU patients from the dataset
2. **Add Manual Patients**: Enter custom patient data for risk analysis
3. **Upload CSV Files**: Bulk analyze multiple patients
4. **View Risk Predictions**: See risk scores from all 7 models
5. **Clinical Scores**: SIRS, qSOFA, NEWS2, SOFA calculations
6. **Explainability**: Understand which features drive predictions

### Model Training (Optional)
If you want to train new models:
```bash
# Run training scripts
python train_with_real_data.py     # Train on real ICU data
python evaluate_models.py           # Evaluate model performance
```

## Setup Instructions

### Local Environment
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run notebooks in order:
```bash
jupyter notebook notebooks/
```

### Google Colab Setup
1. Upload the project to Google Drive
2. Open notebooks in Google Colab
3. Run the setup cell in each notebook to install dependencies
4. Mount Google Drive for data persistence

## Project Structure

```
├── dashboard_real.py           # Main interactive dashboard (Streamlit)
├── models/                     # PyTorch architectures (GRU-D, LSTM, CNN-LSTM, Transformer)
├── explainability_utils/       # Data loading, training, calibration, metrics, explainability, plots
├── surveillance/               # Multi-hospital outbreak-detection layer
├── scripts/                    # Evaluation and doc-conversion helpers (run from project root)
│   └── docs/                  # Markdown -> HTML/PDF converters
├── notebooks/                  # Jupyter notebooks
├── docs/                       # All documentation
│   ├── presentation/          # Presentation scripts and guides
│   ├── PROJECT_SUMMARY.md     # Consolidated status reports
│   └── figures/               # Curves and comparison charts
├── outputs/                    # Model artifacts and results
│   ├── models/               # Trained model weights, scalers, imputers
│   ├── figures/              # Generated visualizations
│   └── results/              # Evaluation results
├── integration_real.py         # Core prediction system (loads all models)
├── dashboard_real.py           # Streamlit dashboard
├── api_service.py              # FastAPI service
├── predict_new_data.py         # CLI prediction
├── train_models.py             # Deep learning training
├── train_baseline_models.py    # Baseline model training
├── Dataset.csv                # ICU patient data
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Notebook Execution Order

1. `01_data_loading_and_exploration.ipynb` - Data pipeline setup
2. `02_model_grud.ipynb` - GRU-D implementation
3. `03_model_lstm.ipynb` - LSTM implementation
4. `04_model_cnn_lstm.ipynb` - CNN-LSTM implementation
5. `05_model_transformer.ipynb` - Transformer implementation
6. `06_model_calibration.ipynb` - Probability calibration
7. `07_comparative_evaluation.ipynb` - Final evaluation and comparison

## Enhanced Features

### Add Patient with Comprehensive Risk Analysis
The dashboard now supports **manual patient entry** with complete clinical risk assessment:

1. **Input All Required Data**: 28 medical parameters including:
   - Demographics (Age, Gender)
   - Vital Signs (HR, BP, Temperature, Respiratory Rate, O2 Saturation)
   - Arterial Blood Gas (pH, PaCO2, SaO2, etc.)
   - Laboratory Values (Lactate, WBC, Platelets, Creatinine, Bilirubin)

2. **Automated Clinical Scoring**:
   - SIRS, qSOFA, NEWS2, and SOFA scores
   - Risk factor identification (8+ clinical indicators)
   - Risk level classification (High/Medium/Low)

3. **Model-Based Predictions**:
   - Uses integration layer to combine predictions from up to 7 models:
     - Baselines: Logistic Regression, Random Forest, XGBoost
     - DL: GRU-D, LSTM, CNN-LSTM, Transformer
   - DL is optional: if no DL checkpoints found in `outputs/models`, only baselines are used
   - Robust fallback to classical model and clinical rules if artifacts unavailable

4. **Intelligent Risk Assessment**:
   - Combines clinical and model predictions
   - Takes conservative approach (maximum of both)
   - Provides actionable clinical recommendations

### Dashboard Capabilities
- **Real-Time Analysis**: Process ICU patient data in real-time
- **Multiple Data Sources**: Real data, uploaded CSV, or manual entry
- **Visual Patient Indicators**: Icons show data source (➕ Manual, 🏥 Real, 📁 Uploaded)
- **Comprehensive Tabs**: Risk Overview, Model Predictions, Risk Trajectory, Explainability, Patient Data
- **Export Functionality**: Export reports, risk data, and performance metrics

## Key Features

- **CPU Optimized**: Designed for CPU training with Google Colab compatibility
- **Reproducible**: Fixed random seeds and documented hyperparameters
- **Modular**: Clean separation of models, utilities, and notebooks
- **Integration Ready**: JSON outputs for dashboard integration

## Success Metrics

- Target AUROC ≥ 0.85
- 4-6 hour prediction lead-time
- Outperform baseline models
- Calibrated probability outputs
- High sensitivity for early detection

## Technical Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with NumPy, Pandas, Scikit-learn
- **Deep Learning (optional)**: PyTorch, used only if DL checkpoints are present
- **Data Preprocessing**: Custom loaders with imputation and scaling
- **Model Inference**: `integration_real.py` combines 3 baselines + 4 DL into an ensemble
- **Fallbacks**: If DL missing → baselines only; if baselines missing → classical model/clinical rules
- **Explainability**: Built-in explainability utilities and narratives

## Dependencies

- Python 3.8+
- Streamlit
- NumPy, Pandas, Scikit-learn, XGBoost
- PyTorch (optional; required only for DL checkpoints)
- See `requirements.txt` for complete list

## Ensemble and Model Artifacts

- Baseline models loaded from:
  - `outputs/models/logistic_regression.pkl`
  - `outputs/models/random_forest.pkl`
  - `outputs/models/xgboost.pkl`
- Deep Learning checkpoints (optional):
  - `outputs/models/grud_demo_model.pt`
  - `outputs/models/lstm_demo_model.pt`
  - `outputs/models/cnn_lstm_demo_model.pt`
  - `outputs/models/transformer_demo_model.pt`
- Dummy DL fallback is disabled. If DL checkpoints are absent, only baselines contribute to the ensemble.

Note: The dashboard imports the integration system from `integration_real.py`. Some files reference an absolute `project_root`; if you move the project folder, update those or replace with `Path(__file__).parent`.

## Contributing

This project represents a collaborative effort with multiple team members contributing different components:
- **Person A**: Preprocessed ICU dataset
- **Person B**: Baseline model results for comparison
- **Person C**: Deep Learning Models Implementation
- **Person D**: Dashboard integration requirements

## License

Academic/Educational Project
