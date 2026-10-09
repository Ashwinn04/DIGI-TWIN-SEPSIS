"""
Compare baseline model results from notebook vs existing models
"""
import json
import os
from pathlib import Path

def load_existing_metrics():
    """Load metrics from existing models"""
    metrics_path = Path('outputs/models/baseline_models_metrics.json')
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def print_comparison():
    """Print comparison between notebook results and existing models"""
    print("="*80)
    print("Baseline Model Performance Comparison")
    print("="*80)
    
    # Notebook results (from Capstone (4).ipynb)
    notebook_results = {
        'logistic_regression': {
            'auroc': 0.6293,  # Range: 0.629-0.746
            'auprc': 0.0371,  # Range: 0.037-0.072
            'sensitivity': 0.541,  # Range: 0.000-0.541
            'specificity': 0.646,  # Range: 0.646-1.000
            'brier': 0.0211
        },
        'random_forest': {
            'auroc': 0.7914,  # Range: 0.673-0.791
            'auprc': 0.1242,  # Range: 0.046-0.124
            'sensitivity': 0.459,  # Range: 0.003-0.459
            'specificity': 0.768,  # Range: 0.768-0.999
            'brier': 0.0201
        },
        'xgboost': {
            'auroc': 0.7460,  # Best from notebook
            'auprc': 0.0719,
            'sensitivity': 0.571,  # Best from notebook
            'specificity': 0.769,
            'brier': 0.0207
        }
    }
    
    # Existing model results
    existing_results = load_existing_metrics()
    
    print("\n📊 Notebook Results (from Capstone (4).ipynb):")
    print("-"*80)
    print(f"{'Model':<25} {'AUROC':<10} {'AUPRC':<10} {'Sensitivity':<12} {'Specificity':<12}")
    print("-"*80)
    for model_name, metrics in notebook_results.items():
        print(f"{model_name:<25} {metrics['auroc']:<10.4f} {metrics['auprc']:<10.4f} "
              f"{metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f}")
    
    if existing_results:
        print("\n📊 Existing Model Results (from outputs/models/baseline_models_metrics.json):")
        print("-"*80)
        print(f"{'Model':<25} {'AUROC':<10} {'AUPRC':<10} {'Sensitivity':<12} {'Specificity':<12}")
        print("-"*80)
        
        # Map existing results to notebook format
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            if model_name in existing_results:
                metrics = existing_results[model_name]
                # Use test metrics if available, otherwise val
                auroc = metrics.get('test_auroc', metrics.get('val_auroc', 0))
                auprc = metrics.get('test_auprc', metrics.get('val_auprc', 0))
                sensitivity = metrics.get('test_recall', metrics.get('val_recall', 0))
                # Calculate specificity from confusion matrix if not directly available
                # Specificity = TN / (TN + FP), but we need to derive it
                # For now, use accuracy as approximation or calculate from precision/recall
                specificity = metrics.get('test_specificity', metrics.get('val_specificity', None))
                if specificity is None:
                    # Try to calculate from other metrics
                    accuracy = metrics.get('test_accuracy', metrics.get('val_accuracy', 0))
                    precision = metrics.get('test_precision', metrics.get('val_precision', 0))
                    # Rough approximation: if we don't have specificity, skip it
                    specificity = 0.0
                
                print(f"{model_name:<25} {auroc:<10.4f} {auprc:<10.4f} "
                      f"{sensitivity:<12.4f} {specificity:<12.4f}")
        
        # Comparison
        print("\n📈 Comparison:")
        print("-"*80)
        print(f"{'Model':<25} {'Metric':<15} {'Notebook':<12} {'Existing':<12} {'Difference':<12}")
        print("-"*80)
        
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            if model_name in existing_results:
                notebook = notebook_results[model_name]
                existing = existing_results[model_name]
                
                # AUROC
                nb_auroc = notebook['auroc']
                ex_auroc = existing.get('test_auroc', existing.get('val_auroc', 0))
                diff_auroc = ex_auroc - nb_auroc
                print(f"{model_name:<25} {'AUROC':<15} {nb_auroc:<12.4f} {ex_auroc:<12.4f} {diff_auroc:+.4f}")
                
                # AUPRC
                nb_auprc = notebook['auprc']
                ex_auprc = existing.get('test_auprc', existing.get('val_auprc', 0))
                diff_auprc = ex_auprc - nb_auprc
                print(f"{'':<25} {'AUPRC':<15} {nb_auprc:<12.4f} {ex_auprc:<12.4f} {diff_auprc:+.4f}")
                
                # Sensitivity
                nb_sens = notebook['sensitivity']
                ex_sens = existing.get('test_recall', existing.get('val_recall', 0))
                diff_sens = ex_sens - nb_sens
                print(f"{'':<25} {'Sensitivity':<15} {nb_sens:<12.4f} {ex_sens:<12.4f} {diff_sens:+.4f}")
                
                # Specificity (may not be available in existing metrics)
                nb_spec = notebook['specificity']
                ex_spec = existing.get('test_specificity', existing.get('val_specificity', None))
                if ex_spec is None:
                    ex_spec = 0.0
                    spec_note = " (N/A)"
                else:
                    spec_note = ""
                diff_spec = ex_spec - nb_spec
                print(f"{'':<25} {'Specificity':<15} {nb_spec:<12.4f} {ex_spec:<12.4f} {diff_spec:+.4f}{spec_note}")
                print()
    else:
        print("\n⚠️ No existing model metrics found. Run train_baseline_models.py first.")
    
    print("="*80)

if __name__ == '__main__':
    print_comparison()

