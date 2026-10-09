"""
Updated Training Script for Real ICU Data
This script trains models using the actual Dataset.csv from Person A
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
from models.grud import create_grud_model, count_grud_parameters
from utils.data_loader import load_preprocessed_data, get_train_val_test_split, create_data_loaders
from utils.training import ModelTrainer, calculate_class_weights, set_random_seeds, get_device, save_training_results, run_training_multiple_seeds
from utils.metrics import calculate_all_metrics, print_metrics_summary
from utils.visualization import plot_training_history, plot_confusion_matrix

def load_real_data():
    """Load and examine the real ICU dataset"""
    print("=== Loading Real ICU Dataset ===")
    
    # Load the dataset
    data_path = os.path.join(project_root, 'Dataset.csv')
    data = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Clean column names and prepare data
    data = data.rename(columns={
        'Hour': 'Time',
        'SepsisLabel': 'Sepsis_Label'
    })
    
    # Get feature columns (exclude metadata columns)
    metadata_cols = ['Unnamed: 0', 'Time', 'Sepsis_Label', 'Patient_ID']
    feature_cols = [col for col in data.columns if col not in metadata_cols]
    
    print(f"\nDataset Statistics:")
    print(f"Total records: {len(data):,}")
    print(f"Unique patients: {data['Patient_ID'].nunique():,}")
    print(f"Time range: {data['Time'].min()} to {data['Time'].max()} hours")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Sepsis cases: {data['Sepsis_Label'].sum():,}")
    print(f"Sepsis rate: {data['Sepsis_Label'].mean():.1%}")
    
    # Check for missing data
    missing_data = data[feature_cols].isnull().sum()
    missing_percent = (missing_data / len(data)) * 100
    print(f"\nMissing Data Analysis:")
    print(f"Features with missing data: {(missing_data > 0).sum()}")
    print(f"Average missing percentage: {missing_percent.mean():.1f}%")
    print(f"Max missing percentage: {missing_percent.max():.1f}%")
    
    return data, feature_cols

def main():
    """Main training function with real data"""
    print("=== Deep Learning Models Training with Real ICU Data ===")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    # Load real data
    data, feature_cols = load_real_data()
    
    # Split data by patient ID to avoid data leakage
    print("\n=== Creating Train/Validation/Test Splits ===")
    train_data, val_data, test_data = get_train_val_test_split(data)
    
    # Create data loaders with real data
    print("\n=== Creating Data Loaders ===")
    batch_size = 32  # Increased for real data
    sequence_length = 24  # 24 hours of data
    prediction_horizon = 4  # Predict 4 hours ahead
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        features=feature_cols,
        add_feature_engineering=True,
        rolling_window=3
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Save configuration for other scripts
    config = {
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'train_patients': len(train_data['Patient_ID'].unique()),
        'val_patients': len(val_data['Patient_ID'].unique()),
        'test_patients': len(test_data['Patient_ID'].unique()),
        'sepsis_rate_train': train_data['Sepsis_Label'].mean(),
        'sepsis_rate_val': val_data['Sepsis_Label'].mean(),
        'sepsis_rate_test': test_data['Sepsis_Label'].mean(),
        'device': device,
        'dataset_info': {
            'total_records': len(data),
            'total_patients': data['Patient_ID'].nunique(),
            'sepsis_cases': data['Sepsis_Label'].sum(),
            'sepsis_rate': data['Sepsis_Label'].mean()
        }
    }
    
    config_path = os.path.join(project_root, 'outputs', 'config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, dict):
            config_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.int64, np.int32)):
                    config_serializable[key][k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    config_serializable[key][k] = float(v)
                else:
                    config_serializable[key][k] = v
        elif isinstance(value, (np.int64, np.int32)):
            config_serializable[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            config_serializable[key] = float(value)
        else:
            config_serializable[key] = value
    
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")
    
    # Train GRU-D model as demonstration
    print("\n=== Training GRU-D Model ===")
    
    # Create model
    # Infer input size from a sample batch to account for engineered features
    sample_batch = next(iter(train_loader))
    inferred_input_size = sample_batch[0].shape[-1]
    model = create_grud_model(
        input_size=inferred_input_size,
        hidden_size=128,  # Increased for real data
        num_layers=2,
        dropout=0.3,
        use_attention=True
    )
    
    print(f"GRU-D model created with {count_grud_parameters(model):,} parameters")
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_loader)
    print(f"Class weights: {class_weights}")

    # Run multi-seed training summary (does not persist models, only aggregates validation performance)
    def grud_factory(input_size: int):
        return create_grud_model(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            use_attention=True
        )
    seeds = [42, 43, 44]
    seeds_result = run_training_multiple_seeds(
        grud_factory, inferred_input_size, seeds,
        train_loader, val_loader, device,
        epochs=20, learning_rate=0.001, weight_decay=1e-5, patience=5,
        use_amp=True, use_focal_loss=True, focal_gamma=2.0
    )
    # Save seeds aggregation summary
    seeds_summary_path = os.path.join(project_root, 'outputs', 'results', 'grud_real_data_seeds_summary.json')
    os.makedirs(os.path.dirname(seeds_summary_path), exist_ok=True)
    with open(seeds_summary_path, 'w') as f:
        json.dump(seeds_result['summary'], f, indent=2)
    print(f"Seeds summary saved to: {seeds_summary_path}")
    
    # Create trainer
    trainer = ModelTrainer(model, device)
    
    # Training parameters
    epochs = 50  # More epochs for real data
    learning_rate = 0.001
    weight_decay = 1e-5
    patience = 15  # More patience for real data
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        patience=patience,
        save_path=os.path.join(project_root, 'outputs', 'models', 'grud_real_data.pt')
    )
    
    # Save training results
    results_path = save_training_results(history, 'grud_real_data', 
                                       os.path.join(project_root, 'outputs', 'results'))
    
    # Plot training history
    plot_training_history(history, 'GRU-D (Real Data)', 
                         os.path.join(project_root, 'outputs', 'figures', 'grud_real_training_history.png'))
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation ===")
    test_loss, test_score = trainer.validate_epoch(test_loader, nn.BCEWithLogitsLoss())
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUROC: {test_score:.4f}")
    
    # Get predictions for detailed evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                features, masks, delta_t, targets = batch
            else:
                features, masks, targets = batch
                delta_t = None
            features = features.to(device)
            masks = masks.to(device)
            if delta_t is not None:
                delta_t = delta_t.to(device)
            targets = targets.to(device)
            
            outputs = model(features, masks, delta_t) if delta_t is not None else model(features, masks)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets_cpu.flatten())
    
    # Calculate metrics
    metrics = calculate_all_metrics(np.array(all_targets), np.array(all_preds))
    print_metrics_summary(metrics, "GRU-D (Real Data)")
    
    # Plot confusion matrix
    plot_confusion_matrix(np.array(all_targets), np.array(all_preds), 
                         "GRU-D (Real Data)", metrics['optimal_threshold'],
                         os.path.join(project_root, 'outputs', 'figures', 'grud_real_confusion_matrix.png'))
    
    print("\n✓ GRU-D model training with real data completed!")
    print(f"✓ Model saved to: {os.path.join(project_root, 'outputs', 'models', 'grud_real_data.pt')}")
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Configuration saved to: {config_path}")
    
    print("\n=== Next Steps ===")
    print("1. Run 'python evaluate_models.py' to evaluate all models with real data")
    print("2. Run 'python integration.py' to create updated integration artifacts")
    print("3. Use the trained models for Person D's dashboard integration")

if __name__ == "__main__":
    main()
