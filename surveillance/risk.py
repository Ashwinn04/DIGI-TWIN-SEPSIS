"""Out-of-fold patient risk scores (every patient is scored by a model that never saw them)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

NON_FEATURES = {"Unnamed: 0", "SepsisLabel", "Patient_ID"}


def patient_risk_scores(csv_path, n_splits=3, seed=42):
    """Return (patient_df, metrics). patient_df: Patient_ID, max_risk, ever_septic.

    Row-level gradient boosting with patient-grouped CV, then max risk over each stay.
    Metrics are reported honestly so they can be quoted (or not) in the paper.
    """
    df = pd.read_csv(csv_path)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X, y, g = df[feats].values, df["SepsisLabel"].values, df["Patient_ID"].values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                           random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    df["risk"] = oof
    pt = df.groupby("Patient_ID").agg(max_risk=("risk", "max"),
                                      ever_septic=("SepsisLabel", "max")).reset_index()
    metrics = {
        "row_level_auroc": float(roc_auc_score(y, oof)),
        "row_level_auprc": float(average_precision_score(y, oof)),
        "patient_level_auroc": float(roc_auc_score(pt.ever_septic, pt.max_risk)),
        "patient_level_auprc": float(average_precision_score(pt.ever_septic, pt.max_risk)),
        "n_patients": int(len(pt)),
        "patient_sepsis_rate": float(pt.ever_septic.mean()),
        "cv": f"{n_splits}-fold patient-grouped, out-of-fold",
    }
    return pt, metrics
