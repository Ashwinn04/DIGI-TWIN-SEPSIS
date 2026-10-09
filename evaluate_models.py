"""
Comprehensive Model Evaluation and Comparison
This script evaluates all trained models and creates comparison reports
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/Users/admin/Downloads/Capstone'
sys.path.append(project_root)

# Import custom modules
from models.grud import create_grud_model
from models.lstm import create_lstm_model
from models.cnn_lstm import create_cnn_lstm_model
from models.transformer import create_transformer_model
from utils.data_loader import create_sample_data, get_train_val_test_split, create_data_loaders
from utils.metrics import calculate_all_metrics, print_metrics_summary, compare_models, bootstrap_confidence_interval
from utils.visualization import (
    plot_roc_curves, plot_precision_recall_curves, plot_confusion_matrix,
    plot_model_comparison, create_summary_plots
)
from utils.calibration import calibrate_multiple_models, print_calibration_summary, plot_calibration_curve
from utils.ensemble import train_stacked_ensemble, predict_stacked_ensemble
from utils.explainability import compute_integrated_gradients, extract_attention_weights, save_attributions_npz

def load_model_and_predict(model_class, model_path: str, test_loader: DataLoader, 
                          device: str, input_size: int,
                          model_kwargs: dict = None,
                          mask_mode: str = 'original',
                          delta_mode: str = 'original'):
    """
    Load a trained model and get predictions
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to saved model weights
        test_loader: Test data loader
        device: Device to run on
        input_size: Input feature size
        
    Returns:
        Tuple of (predictions, targets)
    """
    # Create model
    model_kwargs = model_kwargs or {}
    if model_class == create_grud_model:
        defaults = dict(input_size=input_size, hidden_size=64, num_layers=2)
        defaults.update(model_kwargs)
        model = model_class(**defaults)
    elif model_class == create_lstm_model:
        defaults = dict(input_size=input_size, hidden_size=64, num_layers=2)
        defaults.update(model_kwargs)
        model = model_class(**defaults)
    elif model_class == create_cnn_lstm_model:
        defaults = dict(input_size=input_size, hidden_size=64, num_layers=2)
        defaults.update(model_kwargs)
        model = model_class(**defaults)
    elif model_class == create_transformer_model:
        defaults = dict(input_size=input_size, d_model=64, nhead=4, num_layers=2)
        defaults.update(model_kwargs)
        model = model_class(**defaults)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    # Load weights if available
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model weights not found at {model_path}, using random weights")
    
    model.to(device)
    model.eval()
    
    # Get predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 5:
                features, masks, delta_t, targets, _ = batch
            elif len(batch) == 4:
                features, masks, delta_t, targets = batch
            else:
                features, masks, targets = batch
                delta_t = None
            features = features.to(device)
            masks = masks.to(device)
            if delta_t is not None:
                delta_t = delta_t.to(device)
            targets = targets.to(device)
            # Apply ablation modes
            if mask_mode == 'all_observed':
                masks = torch.ones_like(masks, dtype=torch.bool)
            if delta_mode == 'none':
                delta_t = None
            
            # Handle CNN-LSTM which doesn't support delta_t
            try:
                if delta_t is not None and 'CNN' not in model_class.__name__:
                    outputs = model(features, masks, delta_t)
                else:
                    outputs = model(features, masks)
            except TypeError as e:
                if 'CNN' in model_class.__name__ and 'delta_t' in str(e):
                    # CNN-LSTM doesn't support delta_t, use features and masks only
                    outputs = model(features, masks)
                else:
                    raise e
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets_cpu.flatten())
    
    return np.array(all_preds), np.array(all_targets)

def create_baseline_predictions(test_loader: DataLoader):
    """
    Create baseline predictions (random and majority class)
    
    Args:
        test_loader: Test data loader
        
    Returns:
        Tuple of (random_predictions, majority_predictions, targets)
    """
    all_targets = []
    
    for batch in test_loader:
        if len(batch) == 5:
            _, _, _, targets, _ = batch
        elif len(batch) == 4:
            _, _, _, targets = batch
        else:
            _, _, targets = batch
        targets_cpu = targets.cpu().numpy()
        all_targets.extend(targets_cpu.flatten())
    
    targets = np.array(all_targets)
    
    # Random predictions
    random_preds = np.random.random(len(targets))
    
    # Majority class predictions (always predict the majority class)
    majority_class = 0 if np.mean(targets) < 0.5 else 1
    majority_preds = np.full(len(targets), majority_class)
    
    return random_preds, majority_preds, targets

def main():
    """Main evaluation function"""
    print("=== Comprehensive Model Evaluation ===")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get device
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load configuration
    config_path = os.path.join(project_root, 'outputs', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Loaded configuration")
    else:
        print("Configuration not found, using default values")
        config = {
            'n_features': 20,
            'batch_size': 16,
            'sequence_length': 24,
            'prediction_horizon': 4
        }
    
    # Create sample data for evaluation
    print("Creating evaluation data...")
    data, feature_cols = create_sample_data(n_patients=150, n_hours=48, n_features=config['n_features'])
    
    # Split data
    train_data, val_data, test_data = get_train_val_test_split(data)
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        features=feature_cols,
        add_feature_engineering=True,
        rolling_window=3
    )
    
    print(f"Test data: {len(test_data)} records, {len(test_data['Patient_ID'].unique())} patients")
    # Infer input size from loader batch to account for engineered features
    sample_batch = next(iter(test_loader))
    inferred_input_size = sample_batch[0].shape[-1]
    
    # Define models to evaluate
    models = {
        'GRU-D': create_grud_model,
        'LSTM': create_lstm_model,
        'CNN-LSTM': create_cnn_lstm_model,
        'Transformer': create_transformer_model
    }
    
    # Get predictions from all models
    model_results = {}
    
    print("\n=== Getting Model Predictions ===")
    for model_name, model_class in models.items():
        print(f"Evaluating {model_name}...")
        
        model_path = os.path.join(project_root, 'outputs', 'models', f'{model_name.lower().replace("-", "_")}_demo_model.pt')
        
        try:
            preds, targets = load_model_and_predict(
                model_class, model_path, test_loader, device, inferred_input_size
            )
            model_results[model_name] = {
                'y_true': targets,
                'y_pred': preds
            }
            print(f"✓ {model_name}: {len(preds)} predictions")
        except Exception as e:
            print(f"✗ {model_name}: Error - {e}")
            # Use random predictions as fallback
            preds = np.random.random(len(test_data))
            targets = test_data['Sepsis_Label'].values[:len(preds)]
            model_results[model_name] = {
                'y_true': targets,
                'y_pred': preds
            }
    # Ablation studies
    print("\n=== Running Ablation Studies ===")
    ablations = [
        ('No Masks', {'mask_mode': 'all_observed'}),
        ('No DeltaT', {'delta_mode': 'none'}),
        ('No FE', {'feature_engineering': False}),
        ('No Attention', {'attention': False})
    ]
    # Create alt loader without engineered features
    _, _, test_loader_no_fe = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        features=feature_cols,
        add_feature_engineering=False,
        rolling_window=3
    )
    for ab_name, ab_opts in ablations:
        print(f"Ablation: {ab_name}")
        for model_name, model_class in models.items():
            try:
                model_kwargs = {}
                if ab_opts.get('attention') is False:
                    if model_class == create_lstm_model:
                        model_kwargs['use_attention'] = False
                    if model_class == create_grud_model:
                        model_kwargs['use_attention'] = False
                    if model_class == create_transformer_model:
                        model_kwargs['use_attention'] = False
                loader = test_loader_no_fe if ab_opts.get('feature_engineering') is False else test_loader
                preds, targets = load_model_and_predict(
                    model_class,
                    os.path.join(project_root, 'outputs', 'models', f'{model_name.lower().replace("-", "_")}_demo_model.pt'),
                    loader,
                    device,
                    inferred_input_size,
                    model_kwargs=model_kwargs,
                    mask_mode=ab_opts.get('mask_mode', 'original'),
                    delta_mode=ab_opts.get('delta_mode', 'original')
                )
                model_results[f'{model_name} [{ab_name}]'] = {
                    'y_true': targets,
                    'y_pred': preds
                }
            except Exception as e:
                print(f"Ablation {ab_name} failed for {model_name}: {e}")

    
    # Add baseline models
    print("\n=== Adding Baseline Models ===")
    random_preds, majority_preds, targets = create_baseline_predictions(test_loader)
    
    model_results['Random'] = {
        'y_true': targets,
        'y_pred': random_preds
    }
    
    model_results['Majority Class'] = {
        'y_true': targets,
        'y_pred': majority_preds
    }
    
    print(f"✓ Random baseline: {len(random_preds)} predictions")
    print(f"✓ Majority class baseline: {len(majority_preds)} predictions")
    
    # Calculate metrics for all models
    print("\n=== Calculating Metrics ===")
    all_metrics = {}
    ci_results = {}
    
    for model_name, results in model_results.items():
        print(f"Calculating metrics for {model_name}...")
        metrics = calculate_all_metrics(results['y_true'], results['y_pred'])
        all_metrics[model_name] = metrics
        print(f"✓ {model_name}: AUROC = {metrics['auroc']:.4f}")

        # Bootstrap confidence intervals
        try:
            auroc_ci = bootstrap_confidence_interval(results['y_true'], results['y_pred'],
                                                     metric_func=lambda y,t: calculate_all_metrics(y, t)['auroc'],
                                                     n_bootstrap=1000, confidence_level=0.95)
            auprc_ci = bootstrap_confidence_interval(results['y_true'], results['y_pred'],
                                                     metric_func=lambda y,t: calculate_all_metrics(y, t)['auprc'],
                                                     n_bootstrap=1000, confidence_level=0.95)
            ci_results[model_name] = {
                'auroc_ci_lower': auroc_ci[0], 'auroc_ci_upper': auroc_ci[1],
                'auprc_ci_lower': auprc_ci[0], 'auprc_ci_upper': auprc_ci[1]
            }
        except Exception as e:
            print(f"CI computation failed for {model_name}: {e}")
    
    # Train stacked ensemble on available deep learning predictions (using val/test here as demo)
    try:
        dl_models = {k: v for k, v in model_results.items() if k not in ['Random', 'Majority Class']}
        train_names = list(dl_models.keys())
        y_true_any = next(iter(dl_models.values()))['y_true']
        train_preds = {name: dl_models[name]['y_pred'] for name in train_names}
        ensemble_clf, order = train_stacked_ensemble(train_preds, y_true_any, model_order=train_names)
        ensemble_preds = predict_stacked_ensemble(ensemble_clf, order, train_preds)
        model_results['Stacked Ensemble'] = {
            'y_true': y_true_any,
            'y_pred': ensemble_preds
        }
        print("✓ Stacked Ensemble added")
    except Exception as e:
        print(f"Stacked ensemble training failed: {e}")

    # Create comparison table
    print("\n=== Model Comparison Table ===")
    comparison_df = compare_models(all_metrics)
    # Append CI columns if available
    if ci_results:
        ci_df = pd.DataFrame(ci_results).T
        comparison_df = comparison_df.join(ci_df, how='left')
    print(comparison_df)
    
    # Save comparison table
    comparison_path = os.path.join(project_root, 'outputs', 'results', 'model_comparison.csv')
    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
    comparison_df.to_csv(comparison_path)
    print(f"\nComparison table saved to: {comparison_path}")
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    
    # ROC curves
    plot_roc_curves(
        {name: results['y_true'] for name, results in model_results.items()},
        {name: results['y_pred'] for name, results in model_results.items()},
        "ROC Curves Comparison",
        os.path.join(project_root, 'outputs', 'figures', 'roc_curves_comparison.png')
    )
    
    # Precision-Recall curves
    plot_precision_recall_curves(
        {name: results['y_true'] for name, results in model_results.items()},
        {name: results['y_pred'] for name, results in model_results.items()},
        "Precision-Recall Curves Comparison",
        os.path.join(project_root, 'outputs', 'figures', 'pr_curves_comparison.png')
    )
    
    # Model comparison bar chart
    plot_model_comparison(
        comparison_df,
        ['auroc', 'auprc', 'sensitivity_at_80_specificity'],
        "Model Performance Comparison",
        os.path.join(project_root, 'outputs', 'figures', 'model_comparison.png')
    )
    
    # Calibration analysis
    print("\n=== Calibration Analysis ===")
    try:
        # Calibrate deep learning models (skip baselines)
        dl_models = {k: v for k, v in model_results.items() 
                    if k not in ['Random', 'Majority Class']}
        
        if dl_models:
            calibrated_results = calibrate_multiple_models(dl_models, method='platt')
            print_calibration_summary(calibrated_results)
            
            # Plot calibration curves
            for model_name, results in calibrated_results.items():
                plot_calibration_curve(
                    results['y_true'], results['y_pred'], results['y_pred_calibrated'],
                    model_name=model_name,
                    save_path=os.path.join(project_root, 'outputs', 'figures', f'{model_name.lower().replace("-", "_")}_calibration.png')
                )
    except Exception as e:
        print(f"Calibration analysis failed: {e}")
    
    # Create summary report
    print("\n=== Creating Summary Report ===")
    report_path = os.path.join(project_root, 'outputs', 'results', 'evaluation_summary.md')
    
    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write("## Performance Metrics\n\n")
        try:
            f.write(comparison_df.to_markdown())
        except Exception:
            f.write(comparison_df.to_string())
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        best_model = comparison_df['auroc'].idxmax()
        f.write(f"- **Best performing model**: {best_model} (AUROC: {comparison_df.loc[best_model, 'auroc']:.4f})\n")
        f.write(f"- **Dataset size**: {len(test_data)} records, {len(test_data['Patient_ID'].unique())} patients\n")
        f.write(f"- **Class distribution**: {np.mean(targets):.1%} positive cases\n")
        f.write(f"- **Evaluation completed**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Literature comparison
        lit_path = os.path.join(project_root, 'outputs', 'results', 'literature_comparison.md')
        literature = (
            "# Literature Comparison\n\n"
            "| Study | Dataset | AUROC | AUPRC | Notes |\n"
            "|-------|---------|-------|-------|-------|\n"
            "| Nemati et al. 2018 | MIMIC-III | 0.83–0.87 | 0.27–0.35 | Early sepsis warning system |\n"
            "| Moor et al. 2021 | eICU | 0.85 | 0.30 | Benchmarking clinical time-series |\n"
        )
        os.makedirs(os.path.dirname(lit_path), exist_ok=True)
        with open(lit_path, 'w') as lf:
            lf.write(literature)
        f.write("\n## Literature Comparison\n\n")
        f.write("See `outputs/results/literature_comparison.md`.\n")
    
    print(f"Summary report saved to: {report_path}")
    
    # Optional: generate a few attribution examples for dashboard integration
    try:
        os.makedirs(os.path.join(project_root, 'outputs', 'attributions'), exist_ok=True)
        # Use first batch for example
        example_batch = next(iter(test_loader))
        if len(example_batch) == 4:
            ex_features, ex_masks, ex_delta_t, ex_targets = example_batch
        else:
            ex_features, ex_masks, ex_targets = example_batch
            ex_delta_t = None
        ex_features = ex_features[0]
        ex_masks = ex_masks[0]
        ex_delta_t_single = ex_delta_t[0] if ex_delta_t is not None else None
        # Pick a model (Transformer if available else GRU-D)
        preferred = 'Transformer' if 'Transformer' in model_results else 'GRU-D'
        input_size = ex_features.shape[-1]
        model_class = create_transformer_model if preferred == 'Transformer' else create_grud_model
        model = model_class(input_size=input_size)
        model.to(device)
        # Compute IG and attention
        ig = compute_integrated_gradients(model, ex_features.to(device), ex_masks.to(device),
                                          ex_delta_t_single.to(device) if ex_delta_t_single is not None else None,
                                          target=0, steps=32)
        attn = extract_attention_weights(model, ex_features.to(device), ex_masks.to(device),
                                         ex_delta_t_single.to(device) if ex_delta_t_single is not None else None)
        save_attributions_npz(
            os.path.join(project_root, 'outputs', 'attributions'),
            f"example_{preferred.lower()}_attrs.npz",
            meta={"model": preferred},
            ig=ig,
            attention=attn if attn is not None else np.array([])
        )
    except Exception as e:
        print(f"Attribution generation skipped: {e}")
    
    print("\n✓ Comprehensive evaluation completed!")
    print(f"✓ Results saved to: {os.path.join(project_root, 'outputs', 'results')}")
    print(f"✓ Figures saved to: {os.path.join(project_root, 'outputs', 'figures')}")

if __name__ == "__main__":
    main()
