"""
Visualization utilities for model evaluation and analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import List, Dict, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_roc_curves(y_true_dict: Dict[str, np.ndarray], 
                   y_pred_dict: Dict[str, np.ndarray],
                   title: str = "ROC Curves Comparison",
                   save_path: Optional[str] = None) -> None:
    """
    Plot ROC curves for multiple models
    
    Args:
        y_true_dict: Dictionary with model names as keys and true labels as values
        y_pred_dict: Dictionary with model names as keys and predictions as values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name in y_true_dict.keys():
        y_true = y_true_dict[model_name]
        y_pred = y_pred_dict[model_name]
        
        # Handle NaN values
        valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        if not np.any(valid_mask):
            continue
            
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        fpr, tpr, _ = roc_curve(y_true_clean, y_pred_clean)
        auc_score = np.trapz(tpr, fpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curves(y_true_dict: Dict[str, np.ndarray], 
                                y_pred_dict: Dict[str, np.ndarray],
                                title: str = "Precision-Recall Curves Comparison",
                                save_path: Optional[str] = None) -> None:
    """
    Plot Precision-Recall curves for multiple models
    
    Args:
        y_true_dict: Dictionary with model names as keys and true labels as values
        y_pred_dict: Dictionary with model names as keys and predictions as values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name in y_true_dict.keys():
        y_true = y_true_dict[model_name]
        y_pred = y_pred_dict[model_name]
        
        # Handle NaN values
        valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        if not np.any(valid_mask):
            continue
            
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        precision, recall, _ = precision_recall_curve(y_true_clean, y_pred_clean)
        auc_score = np.trapz(precision, recall)
        
        plt.plot(recall, precision, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    
    # Random classifier baseline
    baseline_precision = np.mean(y_true_dict[list(y_true_dict.keys())[0]])
    plt.axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.5, 
                label=f'Random Classifier (AP = {baseline_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str = "Model",
                         threshold: float = 0.5,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        model_name: Name of the model
        threshold: Classification threshold
        save_path: Path to save the plot
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Sepsis', 'Sepsis'],
                yticklabels=['No Sepsis', 'Sepsis'])
    
    plt.title(f'Confusion Matrix - {model_name}\n(Threshold = {threshold:.3f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict, model_name: str = "Model",
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history (loss and AUROC curves)
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot losses
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot AUROC scores
    ax2.plot(epochs, history['train_scores'], 'b-', label='Training AUROC', linewidth=2)
    ax2.plot(epochs, history['val_scores'], 'r-', label='Validation AUROC', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AUROC', fontsize=12)
    ax2.set_title(f'{model_name} - AUROC Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def plot_risk_trajectories(patient_data: Dict, model_predictions: Dict,
                          patient_ids: List[str], 
                          title: str = "Risk Trajectories",
                          save_path: Optional[str] = None) -> None:
    """
    Plot risk trajectories for sample patients
    
    Args:
        patient_data: Dictionary with patient data
        model_predictions: Dictionary with model predictions
        patient_ids: List of patient IDs to plot
        title: Plot title
        save_path: Path to save the plot
    """
    n_patients = len(patient_ids)
    fig, axes = plt.subplots(n_patients, 1, figsize=(12, 4 * n_patients))
    
    if n_patients == 1:
        axes = [axes]
    
    for i, patient_id in enumerate(patient_ids):
        ax = axes[i]
        
        # Plot sepsis onset
        sepsis_time = patient_data[patient_id]['sepsis_onset']
        if sepsis_time is not None:
            ax.axvline(x=sepsis_time, color='red', linestyle='--', 
                      alpha=0.7, label='Sepsis Onset')
        
        # Plot model predictions
        times = patient_data[patient_id]['times']
        for model_name, predictions in model_predictions.items():
            if patient_id in predictions:
                ax.plot(times, predictions[patient_id], 
                       label=f'{model_name} Risk', linewidth=2)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Risk Score', fontsize=12)
        ax.set_title(f'Patient {patient_id}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Risk trajectories saved to {save_path}")
    
    plt.show()


def plot_calibration_curves(y_true_dict: Dict[str, np.ndarray], 
                           y_pred_dict: Dict[str, np.ndarray],
                           title: str = "Calibration Curves",
                           save_path: Optional[str] = None) -> None:
    """
    Plot calibration curves for multiple models
    
    Args:
        y_true_dict: Dictionary with model names as keys and true labels as values
        y_pred_dict: Dictionary with model names as keys and predictions as values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name in y_true_dict.keys():
        y_true = y_true_dict[model_name]
        y_pred = y_pred_dict[model_name]
        
        # Create calibration curve
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', label=f'{model_name}', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curves saved to {save_path}")
    
    plt.show()


def plot_model_comparison(metrics_df: pd.DataFrame,
                         metrics: List[str] = ['auroc', 'auprc', 'sensitivity_at_80_specificity'],
                         title: str = "Model Performance Comparison",
                         save_path: Optional[str] = None) -> None:
    """
    Plot bar chart comparing model performance
    
    Args:
        metrics_df: DataFrame with model metrics
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            ax = axes[i]
            bars = ax.bar(metrics_df.index, metrics_df[metric], 
                        color=sns.color_palette("husl", len(metrics_df)))
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()


def create_summary_plots(model_results: Dict[str, Dict], 
                        output_dir: str = "outputs/figures") -> None:
    """
    Create all summary plots for model comparison
    
    Args:
        model_results: Dictionary with model results
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    y_true_dict = {}
    y_pred_dict = {}
    
    for model_name, results in model_results.items():
        y_true_dict[model_name] = results['y_true']
        y_pred_dict[model_name] = results['y_pred']
    
    # Create plots
    plot_roc_curves(y_true_dict, y_pred_dict, 
                   save_path=os.path.join(output_dir, "roc_curves.png"))
    
    plot_precision_recall_curves(y_true_dict, y_pred_dict,
                                save_path=os.path.join(output_dir, "pr_curves.png"))
    
    plot_calibration_curves(y_true_dict, y_pred_dict,
                           save_path=os.path.join(output_dir, "calibration_curves.png"))
    
    print(f"All summary plots saved to {output_dir}")


def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray,
                           model_name: str = "Model", top_k: int = 20,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores
        model_name: Name of the model
        top_k: Number of top features to show
        save_path: Path to save the plot
    """
    # Get top k features
    top_indices = np.argsort(importance_scores)[-top_k:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = importance_scores[top_indices]
    
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(top_features)), top_scores, 
                   color=sns.color_palette("viridis", len(top_features)))
    
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'{model_name} - Top {top_k} Feature Importance', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance saved to {save_path}")
    
    plt.show()
