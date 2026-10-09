"""
Stacked ensemble utilities: blends DL model predictions with a meta-learner.
Prefers XGBoost if available, falls back to Logistic Regression.
"""
import numpy as np
from typing import Dict, Tuple, Optional, List


def _try_import_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


def build_meta_features(model_results: Dict[str, np.ndarray], model_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Build meta-feature matrix from a dict of model_name -> predictions.
    Returns (X_meta, feature_names) with columns ordered by model_order (or sorted keys).
    """
    if model_order is None:
        model_order = sorted(model_results.keys())
    cols = []
    feat_names = []
    for name in model_order:
        if name in model_results:
            cols.append(model_results[name].reshape(-1, 1))
            feat_names.append(f"pred_{name}")
    if not cols:
        raise ValueError("No model predictions provided to build meta features")
    X_meta = np.concatenate(cols, axis=1)
    return X_meta, feat_names


def train_stacked_ensemble(train_preds: Dict[str, np.ndarray], y_train: np.ndarray,
                           model_order: Optional[List[str]] = None):
    """
    Train a stacked ensemble meta-learner on model predictions.
    Uses XGBoost if available; otherwise falls back to LogisticRegression.
    Returns fitted model and the order of models used.
    """
    from sklearn.linear_model import LogisticRegression
    X_train, order = build_meta_features(train_preds, model_order)
    xgb = _try_import_xgb()
    if xgb is not None:
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=4,
            verbosity=0
        )
    else:
        clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return clf, order


def predict_stacked_ensemble(model, order: List[str], test_preds: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Predict with a fitted stacked ensemble given the model order and test predictions dict.
    """
    X_test, _ = build_meta_features(test_preds, order)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    # Fallback to decision function / sigmoid
    scores = model.decision_function(X_test)
    # Sigmoid
    return 1.0 / (1.0 + np.exp(-scores))


