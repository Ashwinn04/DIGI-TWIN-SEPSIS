"""
Main training script for all deep learning models
Trains: GRU-D, LSTM, CNN-LSTM, Transformer
"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import GRUD, LSTM, CNNLSTM, Transformer
from explainability_utils.data_loader import (
    load_preprocessed_data,
    create_patient_splits,
    get_train_val_test_split,
    fit_and_save_imputer_scaler,
    apply_imputation_and_scaling,
    create_data_loaders
)
from explainability_utils.training import (
    ModelTrainer,
    get_device,
    set_random_seeds,
    save_training_results
)
from explainability_utils.calibration import ModelCalibrator
from explainability_utils.metrics import find_threshold_for_recall
from explainability_utils.visualization import plot_training_history
import warnings
warnings.filterwarnings('ignore')


# Model configurations
MODEL_CONFIGS = {
    'grud': {
        'class': GRUD,
        'kwargs': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3},
        'save_name': 'grud_real_data.pt'
    },
    'lstm': {
        'class': LSTM,
        'kwargs': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3},
        'save_name': 'lstm_real_data.pt'
    },
    'cnn_lstm': {
        'class': CNNLSTM,
        'kwargs': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.3,
                   'cnn_filters': [64, 128], 'kernel_size': 3},
        'save_name': 'cnn_lstm_real_data.pt'
    },
    'transformer': {
        'class': Transformer,
        'kwargs': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1},
        'save_name': 'transformer_real_data.pt'
    }
}


def train_model_with_seed(model_name: str, seed: int, train_loader, val_loader,
                          input_size: int, device: str, config: dict, args: argparse.Namespace):
    """Train a single model with a specific random seed."""
    print(f"\n--- Training {model_name.upper()} with seed {seed} ---")
    set_random_seeds(seed)

    model_class = config['class']
    model_kwargs = config['kwargs'].copy()
    model_kwargs['input_size'] = input_size
    model = model_class(**model_kwargs)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    trainer = ModelTrainer(model, device=device)

    # Calculate pos_weight
    neg, pos = np.bincount(train_loader.dataset.data['Sepsis_Label'])
    pos_weight = neg / pos if pos > 0 else 1.0

    # Configs for trainer
    optimizer_config = {'name': 'AdamW', 'lr': args.learning_rate, 'weight_decay': 1e-4}
    scheduler_config = {'name': 'CosineAnnealing', 'warmup_epochs': 3, 'T_max': args.epochs - 3}
    loss_config = {
        'name': args.loss_function,
        'pos_weight': pos_weight,
        'label_smoothing': args.label_smoothing if model_name == 'transformer' else 0.0,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0
    }
    early_stopping_config = {'patience': 8, 'metric': 'auprc'}
    
    # Paths for this seed
    base_save_path = os.path.join('outputs', 'models', f"{model_name}_seed{seed}")
    model_save_path = f"{base_save_path}.pt"
    config_save_path = f"{base_save_path}_config.json"
    threshold_save_path = f"{base_save_path}_threshold.json"
    calibrator_save_path = f"{base_save_path}_calibrator.joblib"

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        loss_config=loss_config,
        early_stopping_config=early_stopping_config,
        grad_clip_value=1.0,
        save_path=model_save_path
    )
    
    # --- Post-training evaluation and calibration ---
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Get validation predictions for calibration and thresholding
    val_preds, val_targets = [], []
    with torch.no_grad():
        for features, masks, delta_t, targets in val_loader:
            outputs = model(features.to(device), masks.to(device), delta_t.to(device))
            val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            val_targets.extend(targets.cpu().numpy().flatten())
    
    # Ensure float32 for MPS compatibility
    val_preds = np.array(val_preds, dtype=np.float32)
    val_targets = np.array(val_targets, dtype=np.float32)

    # Find and save threshold for Recall@0.85
    threshold_r85, _, _ = find_threshold_for_recall(val_targets, val_preds, target_recall=0.85)
    with open(threshold_save_path, 'w') as f:
        json.dump({'threshold_r85': threshold_r85}, f)
        
    # Calibrate and save calibrator
    calibrator = ModelCalibrator(method='platt')
    calibrator.fit(val_targets, val_preds)
    calibrator.save(calibrator_save_path)
    
    # Save config
    training_config = {
        'model_name': model_name,
        'seed': seed,
        'input_size': input_size,
        'model_kwargs': model_kwargs,
        'optimizer_config': optimizer_config,
        'scheduler_config': scheduler_config,
        'loss_config': loss_config,
        'sequence_length': args.sequence_length,
        'prediction_horizon': args.prediction_horizon,
        'step_size': args.step_size
    }
    with open(config_save_path, 'w') as f:
        json.dump(training_config, f, indent=2)

    print(f"--- Finished training for seed {seed} ---")
    return {
        'history': history,
        'val_preds': val_preds,
        'val_targets': val_targets
    }


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models for sepsis prediction')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to preprocessed data CSV file')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['grud', 'lstm', 'cnn_lstm', 'transformer', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of training epochs (default: 60)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--sequence_length', type=int, default=48,
                       help='Sequence length in hours (default: 48)')
    parser.add_argument('--prediction_horizon', type=int, default=6,
                       help='Prediction horizon in hours (default: 6)')
    parser.add_argument('--step_size', type=int, default=1,
                       help='Step size for creating sequences in hours (default: 1)')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of seeds to train for ensembling (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--loss_function', type=str, default='BCE', choices=['BCE', 'Focal'],
                       help='Loss function to use (default: BCE)')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                       help='Label smoothing for Transformer (default: 0.05)')

    args = parser.parse_args()
    
    set_random_seeds(args.seed)
    device = get_device()
    
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}")
        data, feature_cols = load_preprocessed_data(args.data_path)
    else:
        print("No data file provided or file not found. Using sample data.")
        from explainability_utils.data_loader import create_sample_data
        data, feature_cols = create_sample_data(n_patients=200, n_hours=72, n_features=40)
    
    input_size = len(feature_cols)
    print(f"Input size: {input_size} features")

    # Patient-level splits
    print("\nCreating or loading patient splits...")
    patient_splits = create_patient_splits(data, seed=args.seed)
    train_data, val_data, test_data = get_train_val_test_split(data, patient_splits)

    # Fit and apply imputer/scaler
    print("\nFitting imputer and scaler on training data...")
    imputer, scaler = fit_and_save_imputer_scaler(train_data, feature_cols)
    
    train_data = apply_imputation_and_scaling(train_data, feature_cols, imputer, scaler)
    val_data = apply_imputation_and_scaling(val_data, feature_cols, imputer, scaler)
    test_data = apply_imputation_and_scaling(test_data, feature_cols, imputer, scaler)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        features=feature_cols,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        step_size=args.step_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Determine which models to train
    models_to_train = []
    if 'all' in args.models:
        models_to_train = ['grud', 'lstm', 'cnn_lstm', 'transformer']
    else:
        models_to_train = args.models
    
    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    
    # --- Main Training Loop ---
    all_results = {}
    
    for model_name in models_to_train:
        model_results_per_seed = []
        for i in range(args.num_seeds):
            seed = args.seed + i
            try:
                results = train_model_with_seed(
                    model_name=model_name,
                    seed=seed,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    input_size=input_size,
                    device=device,
                    config=MODEL_CONFIGS[model_name],
                    args=args
                )
                model_results_per_seed.append(results)
                
            except Exception as e:
                print(f"\n❌ Error training {model_name} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        all_results[model_name] = model_results_per_seed

    # --- Final Summary ---
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    
    final_metrics = []
    for model_name, seed_results in all_results.items():
        if not seed_results: continue
        
        # Ensemble predictions by averaging
        val_preds_ensemble = np.mean([r['val_preds'] for r in seed_results], axis=0)
        val_targets = seed_results[0]['val_targets'] # Targets are the same across seeds
        
        # Calculate ensemble metrics
        from explainability_utils.metrics import calculate_all_metrics
        ensemble_metrics = calculate_all_metrics(val_targets, val_preds_ensemble)
        
        print(f"{model_name.upper()} (Ensemble of {len(seed_results)} seeds):")
        print(f"  Best Val AUPRC (avg across seeds): "
              f"{np.mean([r['history']['best_val_score'] for r in seed_results]):.4f}")
        print(f"  Ensemble Val AUPRC: {ensemble_metrics['auprc']:.4f}")
        print(f"  Ensemble Val AUROC: {ensemble_metrics['auroc']:.4f}")
        
        ensemble_metrics['model'] = model_name
        final_metrics.append(ensemble_metrics)

    # Save final comparison CSV
    if final_metrics:
        df_results = pd.DataFrame(final_metrics)
        df_results.set_index('model', inplace=True)
        csv_path = 'outputs/results/model_comparison.csv'
        df_results.to_csv(csv_path)
        print(f"\nFinal results saved to {csv_path}")
        print(df_results)
    
    print("\n✅ Training pipeline completed!")
    print(f"Models and configs saved to: outputs/models/")
    print(f"Results saved to: outputs/results/")
    print(f"Figures saved to: outputs/figures/")


if __name__ == '__main__':
    main()
