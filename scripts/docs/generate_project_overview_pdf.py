import os
from datetime import datetime

try:
    from fpdf import FPDF
except Exception as e:
    raise SystemExit("fpdf2 not installed. Run: python3 -m pip install fpdf2")


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Sepsis Digital Twin - Project Overview", ln=1, align="C")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R")

    def h1(self, txt):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(20, 20, 20)
        self.ln(2)
        self.multi_cell(190, 7, txt)
        self.ln(2)

    def h2(self, txt):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 30, 30)
        self.ln(1)
        self.multi_cell(190, 6, txt)
        self.ln(1)

    def body(self, txt):
        self.set_font("Helvetica", size=10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(190, 5, txt)

    def bullets(self, items):
        self.set_font("Helvetica", size=10)
        for it in items:
            self.multi_cell(190, 5, f"- {it}")


def build_pdf(output_path: str):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Executive Summary
    pdf.h1("Executive Summary")
    pdf.body(
        "This system predicts sepsis risk 4-6 hours before onset using ICU time-series data. "
        "It combines Person B's tabular baseline (Histogram-based Gradient Boosting, HGB) with Person C's deep learning models "
        "(GRU-D, LSTM, CNN-LSTM, Transformer). Person D delivers the integration layer and dashboard; Person A provides the data pipeline. "
        "The dashboard shows per-model risk, agreement, and an ensemble, and the evaluator produces ROC/PR curves and comparison tables."
    )

    # Architecture
    pdf.h1("System Architecture & Data Flow")
    pdf.h2("High-level flow")
    pdf.bullets([
        "Ingest latest patient data (Dataset.csv) or manual form entry in the dashboard.",
        "Integration layer harmonizes fields, fills missing values, and applies scaling for the baseline.",
        "Baseline path (HGB): feature_list -> imputer -> scaler -> model_final_hgb.predict_proba -> probability.",
        "DL path (GRU-D/LSTM/CNN-LSTM/Transformer): build 24-step sequence & masks -> forward pass -> sigmoid probability.",
        "Dashboard renders individual model risks, agreement, and ensemble; Evaluator produces ROC/PR figures and CSV metrics."
    ])

    pdf.h2("Key repositories & artifacts")
    pdf.bullets([
        "Deep learning checkpoints: outputs/models/{grud_real_data.pt | grud_demo_model.pt, lstm_demo_model.pt, cnn_lstm_demo_model.pt, transformer_demo_model.pt}",
        "Baseline artifacts (Person B): outputs/feature_list.json, outputs/imputer_final.pkl, outputs/scaler_final.pkl, outputs/model_final_hgb.pkl",
        "Evaluation outputs: outputs/figures/*.png, outputs/results/model_comparison.csv, outputs/results/evaluation_summary.md",
        "Dashboard app: dashboard_real.py; Integration layer: integration_real.py; Evaluator: Capstone/evaluate_models.py"
    ])

    # Models
    pdf.h1("Models & Rationale")
    pdf.h2("Baseline (HGB/XGBoost class)")
    pdf.body(
        "A strong tabular baseline for structured ICU data. It handles non-linear interactions, is fast to train/infer, and easy to calibrate. "
        "We use it as the minimum acceptable performance and as a safety fallback when DL checkpoints are missing."
    )
    pdf.h2("Deep Learning (Temporal Modeling)")
    pdf.bullets([
        "GRU-D: decays stale features using masks and time-since-last-seen (delta_t); robust to missingness.",
        "LSTM: recurrent memory for sequential dependencies.",
        "CNN-LSTM: local temporal patterns via convolution, aggregated by LSTM.",
        "Transformer: attention captures long-range dependencies without recurrence."
    ])
    pdf.body(
        "Temporal modeling means learning from trajectories (trends, volatility, timing) rather than single snapshots. "
        "This typically improves early detection at a fixed recall (e.g., 0.85)."
    )

    # Integration details
    pdf.h1("Integration Layer (integration_real.py)")
    pdf.bullets([
        "Provides a single interface for baseline and DL; ensures consistent preprocessing.",
        "GRU-D checkpoint preference matches evaluator (grud_real_data.pt -> grud_demo_model.pt).",
        "On-demand DL inference: builds a 24-step feature sequence; loads state_dict if compatible, otherwise uses default weights to avoid crashes.",
        "Returns a uniform dict: { model_name: { risk_score, risk_level, confidence } } for the dashboard."
    ])

    # Recent enhancements
    pdf.h1("Recent Enhancements Added For Review")
    pdf.bullets([
        "Evaluation saves operating-point metrics at Recall~0.85 (Accuracy@R85, Precision@R85, Specificity@R85, F1@R85, Threshold@R85).",
        "Dashboard Performance tab reads model_comparison.csv and renders Accuracy, AUROC, AUPRC, and @R85 charts.",
        "GRU-D path aligned across evaluator and dashboard: prefer grud_real_data.pt, fallback to grud_demo_model.pt.",
        "HGB baseline integrated end-to-end: feature_list -> imputer -> scaler -> predict_proba.",
        "DL inference made robust: sequence builder + safe state_dict loading so all 4 models attempt to run.",
        "UI polish: removed footer text, modern header/cards, hidden Streamlit default footer/menu."
    ])

    # Evaluation
    pdf.h1("Evaluation & Operating Point")
    pdf.bullets([
        "Evaluator computes AUROC/AUPRC and classification metrics at an optimal threshold.",
        "Also saves metrics at Recall~0.85 (Accuracy@R85, Precision@R85, Specificity@R85, F1@R85, Threshold@R85).",
        "Outputs include ROC/PR curves and a comparison table (CSV)."
    ])

    # Dashboard
    pdf.h1("Dashboard (dashboard_real.py)")
    pdf.bullets([
        "Risk Overview: ensemble score, risk level, agreement metric, recommended action.",
        "Model Predictions: side-by-side DL + HGB risk cards for trust and transparency.",
        "Performance Metrics: renders the latest evaluation CSV and figures (AUROC, AUPRC, @Recall~0.85).",
        "Design: modernized header, clean cards, no marketing footer; production-ready look."
    ])

    # Full functionality walkthrough
    pdf.h1("Full Functionality Walkthrough")
    pdf.h2("Data to Predictions")
    pdf.bullets([
        "Load latest patient record from Dataset.csv or manual form.",
        "Integration harmonizes keys and fills missing values; for baseline it also applies imputer and scaler.",
        "Baseline HGB: strict feature order -> imputer -> scaler -> predict_proba -> risk.",
        "Deep Learning: create 24-step [time x features] with masks; run GRU-D/LSTM/CNN-LSTM/Transformer -> sigmoid -> risk.",
        "Aggregate: show per-model risk, compute ensemble as mean, compute agreement as 1 - std of model risks."
    ])
    pdf.h2("Evaluation and Visualization")
    pdf.bullets([
        "Evaluator computes AUROC, AUPRC, default-threshold metrics, and @Recall~0.85 metrics per model.",
        "Saves ROC/PR plots and model comparison bar chart into outputs/figures; table into outputs/results/model_comparison.csv.",
        "Dashboard reads the CSV to keep visuals aligned with evaluator outputs."
    ])
    pdf.h2("Live Models In Dashboard")
    pdf.bullets([
        "DL used live if checkpoints present: GRU-D, LSTM, CNN-LSTM, Transformer.",
        "Baseline HGB always used if its artifacts exist. Random/Majority remain evaluation-only; they are not used for patient predictions.",
        "If a checkpoint is missing or incompatible, that model is skipped; others still run."
    ])

    # Roles
    pdf.h1("Team Roles & Deliverables")
    pdf.h2("Person A - Data Engineering")
    pdf.bullets([
        "Curated ICU time-series dataset (Dataset.csv) and feature dictionary.",
        "Defined patient-wise splits and missingness strategy.",
        "Provided configuration (outputs/config.json) with feature list and sequence settings."
    ])
    pdf.h2("Person B - Baselines")
    pdf.bullets([
        "Implemented classical ML pipeline (HGB/XGBoost class).",
        "Delivered feature_list.json, imputer_final.pkl, scaler_final.pkl, model_final_hgb.pkl.",
        "Established benchmark metrics and calibration."
    ])
    pdf.h2("Person C - Deep Learning")
    pdf.bullets([
        "Built GRU-D, LSTM, CNN-LSTM, Transformer models and training utilities.",
        "Produced demo/real checkpoints in outputs/models/.",
        "Added evaluation at Recall~0.85 and ablation studies (e.g., No DeltaT)."
    ])
    pdf.h2("Person D - Integration & Dashboard")
    pdf.bullets([
        "Unified integration layer (baseline + DL) for prediction and explainability.",
        "Streamlit dashboard: patient view, model comparison, risk trajectory, performance metrics.",
        "Visual assets (ROC/PR curves, model comparison) and export/reporting."
    ])

    # Operations
    pdf.h1("Operations & Fallbacks")
    pdf.bullets([
        "If a DL checkpoint is missing/incompatible, the model is skipped or runs with default weights (clearly logged).",
        "If baseline artifacts are missing, HGB is skipped; other models still run.",
        "If all models are absent, simple heuristics keep the UI responsive (rare, for demo continuity)."
    ])

    # Current Status & Next Steps
    pdf.h1("Current Status & Next Steps")
    pdf.bullets([
        "Evaluation graphs and CSV are generated; dashboard reads the same checkpoints as the evaluator for GRU-D.",
        "Add or retrain DL checkpoints to improve AUROC/AUPRC and Precision@R85; calibrate and lock production thresholds.",
        "Optional: narrow display to a single primary model + HGB fallback for production, keep multi-model view for validation."
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out = os.path.join(root, "outputs", "docs", "Project_Overview.pdf")
    path = build_pdf(out)
    print(f"PDF written to: {path}")


