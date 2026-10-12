"""
Evaluation metrics for sepsis prediction models
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss, f1_score
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


def calculate_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Area Under ROC Curve"""
    if len(np.unique(y_true)) == 1:
        return 0.5  # Random performance for single class
    
    # Handle NaN values
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if not np.any(valid_mask):
        return 0.5
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(np.unique(y_true_clean)) == 1:
        return 0.5
    
    return roc_auc_score(y_true_clean, y_pred_clean)


def calculate_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Area Under Precision-Recall Curve"""
    # Handle NaN values
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if not np.any(valid_mask):
        return 0.0
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(np.unique(y_true_clean)) == 1:
        return 0.0
    
    return average_precision_score(y_true_clean, y_pred_clean)


def calculate_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Brier Score"""
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if not np.any(valid_mask):
        return 1.0
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    return brier_score_loss(y_true_clean, y_pred_clean)


def find_threshold_for_recall(y_true: np.ndarray, y_pred: np.ndarray, 
                                target_recall: float = 0.85) -> Tuple[float, float, float]:
    """Find the decision threshold that achieves a target recall."""
    # Ensure float32 for MPS compatibility
    y_true = y_true.astype(np.float32) if y_true.dtype != np.float32 else y_true
    y_pred = y_pred.astype(np.float32) if y_pred.dtype != np.float32 else y_pred
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Convert to float32 to avoid float64 issues with MPS
    precision = precision.astype(np.float32)
    recall = recall.astype(np.float32)
    thresholds = thresholds.astype(np.float32)
    
    # The last recall value is 0, and the corresponding threshold is not included.
    # We append a threshold for it to make arrays same length.
    thresholds = np.append(thresholds, np.float32(1.0))
    
    # Find the index of the recall value closest to the target recall
    idx = np.argmin(np.abs(recall - target_recall))
    
    # If multiple thresholds achieve the same recall, choose the one with highest precision
    close_recall_indices = np.where(np.abs(recall - target_recall) < 0.01)[0]
    if len(close_recall_indices) > 0:
        best_idx = close_recall_indices[np.argmax(precision[close_recall_indices])]
    else:
        best_idx = idx

    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def calculate_sensitivity_at_specificity(y_true: np.ndarray, y_pred: np.ndarray, 
                                        target_specificity: float = 0.8) -> float:
    """Calculate sensitivity at a given specificity"""
    # Handle NaN values
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if not np.any(valid_mask):
        return 0.0
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(np.unique(y_true_clean)) == 1:
        return 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true_clean, y_pred_clean)
    
    # Find threshold that achieves target specificity
    target_fpr = 1 - target_specificity
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    return tpr[idx] if idx < len(tpr) else 0.0


def calculate_lead_time(y_true: np.ndarray, y_pred: np.ndarray, 
                       timestamps: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate average lead time (hours before sepsis onset)
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        timestamps: Time stamps for each prediction
        threshold: Classification threshold
        
    Returns:
        Average lead time in hours
    """
    # Find sepsis onset times
    sepsis_onsets = []
    current_patient = None
    onset_time = None
    
    for i, (label, pred, time) in enumerate(zip(y_true, y_pred, timestamps)):
        if label == 1 and onset_time is None:
            onset_time = time
        elif label == 0 and onset_time is not None:
            sepsis_onsets.append(onset_time)
            onset_time = None
    
    if onset_time is not None:
        sepsis_onsets.append(onset_time)
    
    if len(sepsis_onsets) == 0:
        return 0.0
    
    # Calculate lead times for positive predictions
    lead_times = []
    for onset_time in sepsis_onsets:
        # Find predictions before onset that exceed threshold
        before_onset_mask = timestamps < onset_time
        high_risk_mask = y_pred >= threshold
        
        if np.any(before_onset_mask & high_risk_mask):
            # Find the latest prediction before onset
            valid_times = timestamps[before_onset_mask & high_risk_mask]
            latest_prediction = np.max(valid_times)
            lead_time = onset_time - latest_prediction
            lead_times.append(lead_time)
    
    return np.mean(lead_times) if lead_times else 0.0


def calculate_timeliness_score(y_true: np.ndarray, y_pred: np.ndarray,
                              timestamps: np.ndarray, 
                              prediction_horizon: int = 4) -> float:
    """
    Calculate timeliness score based on prediction horizon
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities  
        timestamps: Time stamps
        prediction_horizon: Target prediction horizon in hours
        
    Returns:
        Timeliness score (0-1, higher is better)
    """
    # Find sepsis onset times
    sepsis_onsets = []
    onset_time = None
    
    for i, (label, time) in enumerate(zip(y_true, timestamps)):
        if label == 1 and onset_time is None:
            onset_time = time
        elif label == 0 and onset_time is not None:
            sepsis_onsets.append(onset_time)
            onset_time = None
    
    if onset_time is not None:
        sepsis_onsets.append(onset_time)
    
    if len(sepsis_onsets) == 0:
        return 0.0
    
    timeliness_scores = []
    
    for onset_time in sepsis_onsets:
        # Find predictions in the target horizon window
        horizon_start = onset_time - prediction_horizon
        horizon_end = onset_time
        
        horizon_mask = (timestamps >= horizon_start) & (timestamps < horizon_end)
        
        if np.any(horizon_mask):
            horizon_predictions = y_pred[horizon_mask]
            horizon_times = timestamps[horizon_mask]
            
            # Calculate weighted score based on how early the prediction was made
            weights = (horizon_times - horizon_start) / prediction_horizon
            weighted_score = np.sum(horizon_predictions * weights) / np.sum(weights)
            timeliness_scores.append(weighted_score)
    
    return np.mean(timeliness_scores) if timeliness_scores else 0.0


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         timestamps: np.ndarray = None,
                         prediction_horizon: int = 4) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        timestamps: Time stamps (optional)
        prediction_horizon: Prediction horizon in hours
        
    Returns:
        Dictionary of metrics
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if not np.any(valid_mask):
        return {
            'auroc': 0.5,
            'auprc': 0.0,
            'brier_score': 1.0,
            'sensitivity_at_80_specificity': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f1_score': 0.0,
            'optimal_threshold': 0.5,
            'threshold_at_recall_0.85': 0.5,
            'precision_at_recall_0.85': 0.0,
            'recall_at_recall_0.85': 0.0,
            'f1_at_recall_0.85': 0.0,
            'specificity_at_recall_0.85': 0.0
        }
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    metrics = {}
    
    # Basic metrics
    metrics['auroc'] = calculate_auroc(y_true_clean, y_pred_clean)
    metrics['auprc'] = calculate_auprc(y_true_clean, y_pred_clean)
    metrics['brier_score'] = calculate_brier_score(y_true_clean, y_pred_clean)
    metrics['sensitivity_at_80_specificity'] = calculate_sensitivity_at_specificity(
        y_true_clean, y_pred_clean, target_specificity=0.8)
    
    # Time-based metrics (if timestamps provided)
    if timestamps is not None:
        timestamps_clean = timestamps[valid_mask]
        metrics['lead_time'] = calculate_lead_time(y_true_clean, y_pred_clean, timestamps_clean)
        metrics['timeliness_score'] = calculate_timeliness_score(
            y_true_clean, y_pred_clean, timestamps_clean, prediction_horizon)
    
    # Classification metrics at optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true_clean, y_pred_clean)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_binary = (y_pred_clean >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_clean, y_pred_binary).ravel()
    
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (
        metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    metrics['optimal_threshold'] = optimal_threshold
    
    # Metrics at Recall@0.85
    thresh_r85, precision_r85, recall_r85 = find_threshold_for_recall(
        y_true_clean, y_pred_clean, target_recall=0.85
    )
    y_pred_binary_r85 = (y_pred_clean >= thresh_r85).astype(int)
    
    # Check if confusion matrix can be calculated
    if len(np.unique(y_true_clean)) > 1 and len(np.unique(y_pred_binary_r85)) > 1:
        tn_r85, fp_r85, fn_r85, tp_r85 = confusion_matrix(y_true_clean, y_pred_binary_r85).ravel()
    else: # Handle single-class case
        tn_r85, fp_r85, fn_r85, tp_r85 = 0, 0, 0, 0
        if len(y_true_clean) > 0:
            if np.unique(y_true_clean)[0] == 0: # All negative
                tn_r85 = len(y_true_clean)
            else: # All positive
                tp_r85 = len(y_true_clean)

    metrics['threshold_at_recall_0.85'] = thresh_r85
    metrics['precision_at_recall_0.85'] = precision_r85
    metrics['recall_at_recall_0.85'] = recall_r85
    metrics['f1_at_recall_0.85'] = f1_score(y_true_clean, y_pred_binary_r85)
    metrics['specificity_at_recall_0.85'] = tn_r85 / (tn_r85 + fp_r85) if (tn_r85 + fp_r85) > 0 else 0
    
    return metrics


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                metric_func, n_bootstrap: int = 1000,
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a metric
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        metric_func: Function to calculate metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)
    
    return lower_bound, upper_bound


def print_metrics_summary(metrics: Dict[str, float], model_name: str = "Model"):
    """Print a formatted summary of metrics"""
    print(f"\n{model_name} Performance Summary:")
    print("=" * 50)
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Sensitivity @ 80% Specificity: {metrics['sensitivity_at_80_specificity']:.4f}")
    
    if 'lead_time' in metrics:
        print(f"Average Lead Time: {metrics['lead_time']:.2f} hours")
    if 'timeliness_score' in metrics:
        print(f"Timeliness Score: {metrics['timeliness_score']:.4f}")
    
    print(f"\nClassification Metrics (threshold={metrics['optimal_threshold']:.4f}):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    print(f"\nMetrics at Recall >= 0.85 (threshold={metrics['threshold_at_recall_0.85']:.4f}):")
    print(f"Precision: {metrics['precision_at_recall_0.85']:.4f}")
    print(f"Recall: {metrics['recall_at_recall_0.85']:.4f}")
    print(f"F1-Score: {metrics['f1_at_recall_0.85']:.4f}")
    print(f"Specificity: {metrics['specificity_at_recall_0.85']:.4f}")


def compare_models(model_results: Dict[str, Dict[str, float]]):
    """
    Create comparison table for multiple models
    
    Args:
        model_results: Dictionary with model names as keys and metrics as values
        
    Returns:
        DataFrame with model comparison
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(model_results).T
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    return df
