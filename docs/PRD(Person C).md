# Product Requirements Document (PRD)
### Project: Sepsis Digital Twin Capstone  
### Role: Person C – Deep Learning Models  

---

## 1. Objective
Develop and evaluate advanced deep learning models to predict **sepsis onset 4–6 hours before occurrence** using ICU time-series data.  
The goal is to improve early detection accuracy, timeliness, and interpretability compared to traditional baselines.

---

## 2. Scope
- Implement and optimize **GRU-D**, **LSTM**, **CNN-LSTM**, and **Transformer** architectures.  
- Train and evaluate models using the preprocessed dataset prepared by Person A.  
- Ensure predictions are **timely, calibrated, and clinically relevant**.  
- Provide outputs and documentation for integration with the explainability and dashboard modules (Person D).

---

## 3. Functional Requirements

### 3.1 Model Implementation
- Build reusable PyTorch/TensorFlow modules for:
  - GRU-D (handling missing data and irregular time intervals)
  - LSTM (for sequential ICU data)
  - CNN-LSTM (for local temporal feature extraction)
  - Transformer (for attention-based modeling)
- Implement early-warning prediction pipelines for 4h and 6h horizons.

### 3.2 Data Handling
- Input format: Hourly ICU data grouped by **Patient_ID**.  
- Sequence length, padding, and masking handled automatically.  
- Use the training, validation, and test splits defined by Person A.  

### 3.3 Training & Calibration
- Optimize using **Adam optimizer** with learning-rate scheduling.  
- Employ **early stopping** based on validation AUROC.  
- Calibrate probability outputs using **Platt scaling** or **isotonic regression**.  
- Log all experiments, metrics, and hyperparameters.

### 3.4 Evaluation Metrics
- AUROC, AUPRC  
- Sensitivity @ 80% specificity  
- Lead-time (average hours before onset predicted)  
- Timeliness score

---

## 4. Deliverables

| Deliverable | Description |
|--------------|-------------|
| Trained Models | GRU-D, LSTM, CNN-LSTM, Transformer models with saved weights. |
| Training Scripts | Clean, modular code notebooks for model training, validation, and testing. |
| Evaluation Report | Comparative table of performance metrics (AUROC, AUPRC, Timeliness). |
| Visualizations | Risk trajectory plots, learning curves, and confusion matrices. |
| Documentation | Model architecture diagrams, hyperparameter tables, and tuning notes. |
| Integration Artifacts | Serialized model outputs and JSON summaries for Person D’s dashboard integration. |

---

## 5. Non-Functional Requirements
- **Reproducibility:** Random seed control and environment specification (`requirements.txt`).  
- **Scalability:** Modular code structure for future model additions.  
- **Efficiency:** Training optimized for GPU (batching, mixed precision).  
- **Interpretability Support:** Enable feature importance hooks for SHAP/LIME integration.

---

## 6. Dependencies
- **Person A:** Cleaned and preprocessed dataset.  
- **Person B:** Baseline metrics for comparison.  
- **Person D:** Dashboard integration and explainability modules.  

---

## 7. Timeline (Milestones)

| Phase | Deliverables | Deadline |
|-------|---------------|-----------|
| Phase 1 | GRU-D & LSTM prototypes trained | August |
| Phase 2 | CNN-LSTM & Transformer models, calibration | September/October |
| Phase 3 | Final comparative analysis, documentation, integration-ready outputs | November |
| Phase 4 | Report & presentation materials | December |

---

## 8. Success Criteria
- ≥ **0.85 AUROC** and improvement over baseline models.  
- Predictive lead-time ≥ **4 hours before sepsis onset**.  
- Clear documentation of architecture, tuning, and results.  
- Seamless integration with explainability dashboard.
