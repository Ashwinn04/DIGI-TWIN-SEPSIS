> **Team** **Playbook:** **Sepsis** **Digital** **Twin** **Capstone**
> **Project**

**1.** **Data** **Handling** **Rules**

\- Source: Dataset.csv (hourly ICU data, ~546k rows, 44 cols).

\- ID Structure: Use Patient_ID to group records. Always split data by
patient, not by row, to avoid leakage.

\- Splits: Train 70%, Validation 15%, Test 15% (by patient).

\- Missing Data: LOCF for vitals, medians for labs, add mask vars. -
Normalization: Scale features (z-score/min-max), save scalers.

**2.** **Model** **Development** **Standards**

\- Baselines: SIRS, qSOFA, NEWS, SOFA + Logistic Regression, RF,
XGBoost. - Advanced Models: LSTM, GRU-D, Transformer.

\- Horizon: Predict sepsis 4–6h before onset.

\- Evaluation: AUROC, AUPRC, Sensitivity @ 80% specificity, timeliness.

**3.** **Explainability** **&** **Trust**

\- Post-hoc Explainability: SHAP, LIME, Integrated Gradients. - Top 5
contributing features per patient prediction.

\- Report in clinical terms: Rising HR, Low MAP, High Lactate.

**4.** **Validation** **&** **Reporting**

\- Primary Dataset: Dataset.csv. External optional: PhysioNet 2019,
eICU. - Always compare baseline vs ML vs DL.

\- Visualize ROC, PR, calibration, trajectories. - Document
preprocessing & hyperparams.

**5.** **Collaboration** **Workflow**

\- Version Control: GitHub branches per member.

\- Code: Python (Pandas, Sklearn, PyTorch/TF, SHAP). - Meet weekly,
deliver notebooks + outputs.

\- Shared doc for report, shared folder for figures.

**6.** **Milestone** **Expectations**

\- First Review (Aug): Dataset cleaned, baseline scoring, DL plan,
dashboard mockup. - Second Review (Sep/Oct): Baseline ready, DL results,
SHAP prototypes.

\- Final Review (Nov): Mature DL, explainability, dashboard,
documentation. - Final Report (Dec): IEEE paper, capstone report, demo
video.

**7.** **Golden** **Rules**

\- Never mix patient records across train/test. - Always compare against
baselines.

\- Check AUROC + PR + timeliness. - Document all experiments.

\- Accuracy + Explainability = Success.

> **Person-wise** **Roles** **and** **Responsibilities**
>
> **Person** **A** **–** **Data** **Engineering** **&** **Management**

***Duties:***

• Clean and preprocess dataset (handle NaNs, resample hourly).

• Perform feature engineering (rolling stats, SOFA components, shock
index). • Maintain consistent train/validation/test patient splits.

• Provide dataset documentation (stats, distributions, missingness
analysis).

***Deliverables:***

• Final cleaned dataset with preprocessing scripts. • Feature
engineering notebook.

• Descriptive cohort statistics and visualizations.

> **Person** **B** **–** **Baselines** **&** **Clinical** **Scores**

***Duties:***

• Implement clinical scoring systems (SIRS, qSOFA, NEWS, SOFA).

• Develop baseline ML models (Logistic Regression, Random Forest,
XGBoost). • Conduct ablation studies (vitals only, labs only).

• Benchmark models against clinical scores.

***Deliverables:***

• Baseline model performance tables (AUROC, AUPRC, Sensitivity,
Specificity). • Comparative analysis report.

• Figures: ROC/PR curves, calibration plots.

> **Person** **C** **–** **Deep** **Learning** **Models**

***Duties:***

• Implement advanced models: GRU-D, LSTM, CNN-LSTM, Transformer. • Train
models for different prediction horizons (4–6h before onset).

• Calibrate predictions and measure timeliness.

• Document model architectures and hyperparameters.

***Deliverables:***

• Trained deep learning models and scripts.

• Risk trajectory plots and lead-time evaluation.

• Report section: Novelty and deep model performance.

> **Person** **D** **–** **Explainability** **&** **Dashboard**

***Duties:***

• Apply explainability tools: SHAP, LIME, Integrated Gradients. • Build
Streamlit/FastAPI dashboard to simulate the digital twin. • Integrate
model predictions with interpretability outputs.

• Prepare presentation/demo materials.

***Deliverables:***

• Interactive digital twin dashboard.

• Explainability reports and feature attribution graphs. • Demo
video/screenshots for final presentation.
