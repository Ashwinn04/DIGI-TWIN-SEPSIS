"""
Calculate and add accuracy to baseline models metrics JSON file.
Accuracy = (TP + TN) / (TP + TN + FP + FN)
We can derive this from sensitivity, specificity, and the confusion matrix components.
"""
import json
import os
from pathlib import Path

def calculate_accuracy_from_metrics(sensitivity, specificity, prevalence=None):
    """
    Calculate accuracy from sensitivity and specificity.
    
    If we know the test set prevalence, we can calculate:
    Accuracy = sensitivity * prevalence + specificity * (1 - prevalence)
    
    However, we need the actual confusion matrix to get exact accuracy.
    This function provides an approximation if prevalence is known.
    """
    if prevalence is None:
        # Use a typical test set prevalence (from the notebook, it was ~2.17%)
        prevalence = 0.0217
    
    accuracy = sensitivity * prevalence + specificity * (1 - prevalence)
    return accuracy

def add_accuracy_to_metrics():
    """Add accuracy field to baseline_models_metrics.json"""
    
    metrics_paths = [
        'outputs/models/baseline_models_metrics.json',
        'Capstone/outputs/models/baseline_models_metrics.json'
    ]
    
    for metrics_path in metrics_paths:
        if not os.path.exists(metrics_path):
            continue
        
        print(f"📂 Loading metrics from: {metrics_path}")
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Calculate accuracy for each model
            # From the notebook output, we know:
            # - Logistic Regression: Accuracy = 0.6402 (64.02%)
            # - Random Forest: Accuracy = 0.9750 (97.50%)
            # - XGBoost: Accuracy = 0.9782 (97.82%)
            
            # These are from the notebook evaluation
            known_accuracies = {
                'logistic_regression': 0.6402,
                'random_forest': 0.9750,
                'xgboost': 0.9782
            }
            
            updated = False
            for model_name in metrics.keys():
                if 'accuracy' not in metrics[model_name]:
                    # Use known accuracy if available, otherwise calculate
                    if model_name in known_accuracies:
                        metrics[model_name]['accuracy'] = known_accuracies[model_name]
                        print(f"✅ Added accuracy for {model_name}: {known_accuracies[model_name]:.4f}")
                    else:
                        # Calculate from sensitivity and specificity
                        sens = metrics[model_name].get('sensitivity', 0)
                        spec = metrics[model_name].get('specificity', 0)
                        if sens > 0 or spec > 0:
                            acc = calculate_accuracy_from_metrics(sens, spec)
                            metrics[model_name]['accuracy'] = acc
                            print(f"✅ Calculated accuracy for {model_name}: {acc:.4f}")
                    updated = True
            
            if updated:
                # Backup original file
                backup_path = metrics_path + '.backup'
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy(metrics_path, backup_path)
                    print(f"💾 Created backup: {backup_path}")
                
                # Save updated metrics
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"✅ Updated {metrics_path} with accuracy values")
                return True
            else:
                print(f"ℹ️ {metrics_path} already has accuracy values")
                return True
                
        except Exception as e:
            print(f"❌ Error processing {metrics_path}: {e}")
            continue
    
    print("⚠️ No baseline_models_metrics.json file found")
    return False

if __name__ == "__main__":
    print("="*60)
    print("Adding Accuracy to Baseline Models Metrics")
    print("="*60)
    add_accuracy_to_metrics()
    print("\n✅ Done!")

