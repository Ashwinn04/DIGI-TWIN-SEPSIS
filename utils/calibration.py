"""
Model Calibration Utilities
Apply Platt scaling and isotonic regression to calibrate model probabilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelCalibrator:
    """
    Calibrate model probabilities using Platt scaling or isotonic regression
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize calibrator
        
        Args:
            method: 'platt' for Platt scaling or 'isotonic' for isotonic regression
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> 'ModelCalibrator':
        """
        Fit the calibrator
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            
        Returns:
            Self for method chaining
        """
        if self.method == 'platt':
            # Platt scaling using logistic regression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_pred.reshape(-1, 1), y_true)
        elif self.method == 'isotonic':
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Calibrate predictions
        
        Args:
            y_pred: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before making predictions")
        
        if self.method == 'platt':
            return self.calibrator.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            return self.calibrator.predict(y_pred)
    
    def get_calibration_error(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for calibration error calculation
            
        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def calibrate_model_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                               method: str = 'platt') -> Tuple[np.ndarray, ModelCalibrator]:
    """
    Calibrate model predictions
    
    Args:
        y_true: True binary labels
        y_pred: Raw predicted probabilities
        method: Calibration method ('platt' or 'isotonic')
        
    Returns:
        Tuple of (calibrated_predictions, calibrator)
    """
    calibrator = ModelCalibrator(method=method)
    calibrator.fit(y_true, y_pred)
    calibrated_preds = calibrator.predict(y_pred)
    
    return calibrated_preds, calibrator


def evaluate_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                        calibrated_pred: np.ndarray = None) -> Dict[str, float]:
    """
    Evaluate calibration quality
    
    Args:
        y_true: True binary labels
        y_pred: Raw predicted probabilities
        calibrated_pred: Calibrated predictions (optional)
        
    Returns:
        Dictionary of calibration metrics
    """
    metrics = {}
    
    # Brier score (lower is better)
    metrics['brier_score'] = brier_score_loss(y_true, y_pred)
    
    # Log loss (lower is better)
    metrics['log_loss'] = log_loss(y_true, y_pred)
    
    # Expected Calibration Error (lower is better)
    calibrator = ModelCalibrator()
    metrics['ece'] = calibrator.get_calibration_error(y_true, y_pred)
    
    if calibrated_pred is not None:
        metrics['brier_score_calibrated'] = brier_score_loss(y_true, calibrated_pred)
        metrics['log_loss_calibrated'] = log_loss(y_true, calibrated_pred)
        metrics['ece_calibrated'] = calibrator.get_calibration_error(y_true, calibrated_pred)
    
    return metrics


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray,
                          calibrated_pred: np.ndarray = None,
                          model_name: str = "Model",
                          n_bins: int = 10,
                          save_path: Optional[str] = None) -> None:
    """
    Plot calibration curve
    
    Args:
        y_true: True binary labels
        y_pred: Raw predicted probabilities
        calibrated_pred: Calibrated predictions (optional)
        model_name: Name of the model
        n_bins: Number of bins for calibration plot
        save_path: Path to save the plot
    """
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(10, 8))
    
    # Plot original predictions
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=n_bins)
    
    plt.plot(mean_predicted_value, fraction_of_positives, 
             marker='o', linewidth=2, label=f'{model_name} (Original)')
    
    # Plot calibrated predictions if provided
    if calibrated_pred is not None:
        fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
            y_true, calibrated_pred, n_bins=n_bins)
        
        plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, 
                 marker='s', linewidth=2, label=f'{model_name} (Calibrated)')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'{model_name} - Calibration Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    
    plt.show()


def calibrate_multiple_models(model_results: Dict[str, Dict],
                             method: str = 'platt') -> Dict[str, Dict]:
    """
    Calibrate predictions for multiple models
    
    Args:
        model_results: Dictionary with model names as keys and results as values
                      Each result should contain 'y_true' and 'y_pred'
        method: Calibration method ('platt' or 'isotonic')
        
    Returns:
        Dictionary with calibrated results
    """
    calibrated_results = {}
    
    for model_name, results in model_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        # Calibrate predictions
        calibrated_pred, calibrator = calibrate_model_predictions(
            y_true, y_pred, method=method)
        
        # Evaluate calibration
        calibration_metrics = evaluate_calibration(y_true, y_pred, calibrated_pred)
        
        calibrated_results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_calibrated': calibrated_pred,
            'calibrator': calibrator,
            'calibration_metrics': calibration_metrics
        }
    
    return calibrated_results


def print_calibration_summary(calibrated_results: Dict[str, Dict]) -> None:
    """
    Print calibration summary for all models
    
    Args:
        calibrated_results: Results from calibrate_multiple_models
    """
    print("=== Model Calibration Summary ===")
    print()
    
    for model_name, results in calibrated_results.items():
        metrics = results['calibration_metrics']
        
        print(f"{model_name}:")
        print(f"  Original ECE: {metrics['ece']:.4f}")
        print(f"  Calibrated ECE: {metrics['ece_calibrated']:.4f}")
        print(f"  Original Brier Score: {metrics['brier_score']:.4f}")
        print(f"  Calibrated Brier Score: {metrics['brier_score_calibrated']:.4f}")
        print(f"  Original Log Loss: {metrics['log_loss']:.4f}")
        print(f"  Calibrated Log Loss: {metrics['log_loss_calibrated']:.4f}")
        print()


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Generate poorly calibrated predictions
    y_pred = np.random.beta(2, 5, n_samples)
    y_pred[y_true == 1] = np.random.beta(5, 2, np.sum(y_true))
    
    # Calibrate predictions
    calibrated_pred, calibrator = calibrate_model_predictions(y_true, y_pred)
    
    # Evaluate calibration
    metrics = evaluate_calibration(y_true, y_pred, calibrated_pred)
    
    print("Calibration Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot calibration curve
    plot_calibration_curve(y_true, y_pred, calibrated_pred, "Sample Model")
