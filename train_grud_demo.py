"""
Simple GRU-D Model Training Script
This script demonstrates training the GRU-D model for sepsis prediction
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
from utils.data_loader import create_sample_data, get_train_val_test_split, create_data_loaders
from utils.training import ModelTrainer, calculate_class_weights, set_random_seeds, get_device, save_training_results
from utils.metrics import calculate_all_metrics, print_metrics_summary
from utils.visualization import plot_training_history, plot_confusion_matrix

def main():
    """Main training function"""
    print("=== GRU-D Model Training Demo ===")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    # Create sample data for demonstration
    print("Creating sample data...")
    data, feature_cols = create_sample_data(n_patients=100, n_hours=48, n_features=20)
    
    # Split data
    train_data, val_data, test_data = get_train_val_test_split(data)
    
    # Create data loaders
    batch_size = 16  # Smaller batch size for CPU
    sequence_length = 24
    prediction_horizon = 4
    
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
    
    # Create model
    # Infer input size from a sample batch to account for engineered features
    sample_batch = next(iter(train_loader))
    inferred_input_size = sample_batch[0].shape[-1]
    model = create_grud_model(
        input_size=inferred_input_size,
        hidden_size=64,  # Smaller for CPU
        num_layers=2,
        dropout=0.3,
        use_attention=True
    )
    
    print(f"Model created with {count_grud_parameters(model):,} parameters")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader)
    print(f"Class weights: {class_weights}")
    
    # Create trainer
    trainer = ModelTrainer(model, device)
    
    # Training parameters
    epochs = 20  # Fewer epochs for demo
    learning_rate = 0.001
    weight_decay = 1e-5
    patience = 5
    
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
        save_path=os.path.join(project_root, 'outputs', 'models', 'grud_demo_model.pt')
    )
    
    # Save training results
    results_path = save_training_results(history, 'grud_demo', os.path.join(project_root, 'outputs', 'results'))
    
    # Plot training history
    plot_training_history(history, 'GRU-D Demo', 
                         os.path.join(project_root, 'outputs', 'figures', 'grud_demo_training_history.png'))
    
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
    print_metrics_summary(metrics, "GRU-D Demo")
    
    # Plot confusion matrix
    plot_confusion_matrix(np.array(all_targets), np.array(all_preds), 
                         "GRU-D Demo", metrics['optimal_threshold'],
                         os.path.join(project_root, 'outputs', 'figures', 'grud_demo_confusion_matrix.png'))
    
    print("\n✓ GRU-D model training demo completed!")
    print(f"✓ Model saved to: {os.path.join(project_root, 'outputs', 'models', 'grud_demo_model.pt')}")
    print(f"✓ Results saved to: {results_path}")
    
    # Save configuration for future use
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
        'device': device
    }
    
    config_path = os.path.join(project_root, 'outputs', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration saved to: {config_path}")

if __name__ == "__main__":
    main()
