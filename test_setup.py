"""
Test script to verify the setup
"""
import sys
import os

# Add project root to path
project_root = '/Users/admin/Downloads/Capstone'
sys.path.append(project_root)

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from utils.data_loader import load_preprocessed_data, create_sample_data
        print("✓ utils.data_loader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils.data_loader: {e}")
        return False
    
    try:
        from utils.metrics import calculate_all_metrics
        print("✓ utils.metrics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils.metrics: {e}")
        return False
    
    try:
        from utils.training import ModelTrainer, set_random_seeds
        print("✓ utils.training imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils.training: {e}")
        return False
    
    try:
        from utils.visualization import plot_training_history
        print("✓ utils.visualization imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils.visualization: {e}")
        return False
    
    try:
        from models.grud import create_grud_model
        print("✓ models.grud imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models.grud: {e}")
        return False
    
    try:
        from models.lstm import create_lstm_model
        print("✓ models.lstm imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models.lstm: {e}")
        return False
    
    try:
        from models.cnn_lstm import create_cnn_lstm_model
        print("✓ models.cnn_lstm imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models.cnn_lstm: {e}")
        return False
    
    try:
        from models.transformer import create_transformer_model
        print("✓ models.transformer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models.transformer: {e}")
        return False
    
    return True

def test_sample_data():
    """Test sample data creation"""
    print("\nTesting sample data creation...")
    
    try:
        from utils.data_loader import create_sample_data, get_train_val_test_split, create_data_loaders
        
        # Create sample data
        data, feature_cols = create_sample_data(n_patients=50, n_hours=24, n_features=10)
        print(f"✓ Sample data created: {data.shape}")
        
        # Split data
        train_data, val_data, test_data = get_train_val_test_split(data)
        print(f"✓ Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data, batch_size=16, sequence_length=12, prediction_horizon=4, features=feature_cols
        )
        print(f"✓ Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def test_models():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        import torch
        
        # Test GRU-D model
        from models.grud import create_grud_model, count_grud_parameters
        grud_model = create_grud_model(input_size=10, hidden_size=64)
        param_count = count_grud_parameters(grud_model)
        print(f"✓ GRU-D model created with {param_count:,} parameters")
        
        # Test LSTM model
        from models.lstm import create_lstm_model, count_lstm_parameters
        lstm_model = create_lstm_model(input_size=10, hidden_size=64)
        param_count = count_lstm_parameters(lstm_model)
        print(f"✓ LSTM model created with {param_count:,} parameters")
        
        # Test CNN-LSTM model
        from models.cnn_lstm import create_cnn_lstm_model, count_cnn_lstm_parameters
        cnn_lstm_model = create_cnn_lstm_model(input_size=10, hidden_size=64)
        param_count = count_cnn_lstm_parameters(cnn_lstm_model)
        print(f"✓ CNN-LSTM model created with {param_count:,} parameters")
        
        # Test Transformer model
        from models.transformer import create_transformer_model, count_transformer_parameters
        transformer_model = create_transformer_model(input_size=10, d_model=64)
        param_count = count_transformer_parameters(transformer_model)
        print(f"✓ Transformer model created with {param_count:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_forward_pass():
    """Test model forward pass"""
    print("\nTesting forward pass...")
    
    try:
        import torch
        
        # Create sample data
        batch_size = 4
        seq_len = 12
        input_size = 10
        
        x = torch.randn(batch_size, seq_len, input_size)
        mask = torch.ones(batch_size, seq_len, input_size)
        
        # Test GRU-D
        from models.grud import create_grud_model
        grud_model = create_grud_model(input_size=input_size, hidden_size=32)
        output = grud_model(x, mask)
        print(f"✓ GRU-D forward pass: {x.shape} -> {output.shape}")
        
        # Test LSTM
        from models.lstm import create_lstm_model
        lstm_model = create_lstm_model(input_size=input_size, hidden_size=32)
        output = lstm_model(x, mask)
        print(f"✓ LSTM forward pass: {x.shape} -> {output.shape}")
        
        # Test CNN-LSTM
        from models.cnn_lstm import create_cnn_lstm_model
        cnn_lstm_model = create_cnn_lstm_model(input_size=input_size, hidden_size=32)
        output = cnn_lstm_model(x, mask)
        print(f"✓ CNN-LSTM forward pass: {x.shape} -> {output.shape}")
        
        # Test Transformer
        from models.transformer import create_transformer_model
        transformer_model = create_transformer_model(input_size=input_size, d_model=32)
        output = transformer_model(x, mask)
        print(f"✓ Transformer forward pass: {x.shape} -> {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Deep Learning Setup ===\n")
    
    tests = [
        test_imports,
        test_sample_data,
        test_models,
        test_forward_pass
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready for training.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()
