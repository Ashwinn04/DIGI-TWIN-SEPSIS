"""
GRU-D Model Training Script
This script trains the GRU-D model for sepsis prediction
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
from utils.training import ModelTrainer, calculate_class_weights, set_random_seeds, get_device, save_training_results
from utils.metrics import calculate_all_metrics, print_metrics_summary
from utils.visualization import plot_training_history, plot_confusion_matrix

def main():
    """Main training function"""
    print("=== GRU-D Model Training ===")
    
    # Set random seeds
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    # Load configuration
    config_path = os.path.join(project_root, 'outputs', 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Loaded configuration from previous run")
    else:
        print("Configuration not found. Please run 01_data_loading_and_exploration.ipynb first")
        return
    
    # Load data
    data_path = os.path.join(project_root, 'Dataset.csv')
    try:
        data, feature_cols = load_preprocessed_data(data_path)
        print(f"Loaded data with {len(feature_cols)} features")
    except FileNotFoundError:
        print("Data not found. Creating sample data...")
        data, feature_cols = create_sample_data(n_patients=200, n_hours=48, n_features=20)
    
    # Split data
    train_data, val_data, test_data = get_train_val_test_split(data)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        prediction_horizon=config['prediction_horizon'],
        features=feature_cols
    )
    
    # Create model
    model = create_grud_model(
        input_size=len(feature_cols),
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model created with {count_grud_parameters(model):,} parameters")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_loader)
    print(f"Class weights: {class_weights}")
    
    # Create trainer
    trainer = ModelTrainer(model, device)
    
    # Training parameters
    epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-5
    patience = 10
    
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
        save_path=os.path.join(project_root, 'outputs', 'models', 'grud_model.pt')
    )
    
    # Save training results
    results_path = save_training_results(history, 'grud', os.path.join(project_root, 'outputs', 'results'))
    
    # Plot training history
    plot_training_history(history, 'GRU-D', 
                         os.path.join(project_root, 'outputs', 'figures', 'grud_training_history.png'))
    
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
        for features, masks, targets in test_loader:
            features = features.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            outputs = model(features, masks)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets_cpu.flatten())
    
    # Calculate metrics
    metrics = calculate_all_metrics(np.array(all_targets), np.array(all_preds))
    print_metrics_summary(metrics, "GRU-D")
    
    # Plot confusion matrix
    plot_confusion_matrix(np.array(all_targets), np.array(all_preds), 
                         "GRU-D", metrics['optimal_threshold'],
                         os.path.join(project_root, 'outputs', 'figures', 'grud_confusion_matrix.png'))
    
    print("\n✓ GRU-D model training completed!")
    print(f"✓ Model saved to: {os.path.join(project_root, 'outputs', 'models', 'grud_model.pt')}")
    print(f"✓ Results saved to: {results_path}")

if __name__ == "__main__":
    main()
