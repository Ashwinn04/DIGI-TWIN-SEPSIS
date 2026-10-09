"""
Simple API for Sepsis Prediction Dashboard Integration
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/Users/admin/Downloads/Capstone'
sys.path.append(project_root)

from integration import ModelInference

app = Flask(__name__)

# Initialize models
models = {}
model_configs = {
  "grud": {
    "model_path": "/Users/admin/Downloads/Capstone/outputs/models/grud_demo_model.pt",
    "model_type": "grud",
    "input_size": 20
  },
  "lstm": {
    "model_path": "/Users/admin/Downloads/Capstone/outputs/models/lstm_demo_model.pt",
    "model_type": "lstm",
    "input_size": 20
  },
  "cnn_lstm": {
    "model_path": "/Users/admin/Downloads/Capstone/outputs/models/cnn_lstm_demo_model.pt",
    "model_type": "cnn_lstm",
    "input_size": 20
  },
  "transformer": {
    "model_path": "/Users/admin/Downloads/Capstone/outputs/models/transformer_demo_model.pt",
    "model_type": "transformer",
    "input_size": 20
  }
}

for model_name, config in model_configs.items():
    try:
        models[model_name] = ModelInference(
            model_path=config['model_path'],
            model_type=config['model_type'],
            input_size=config['input_size']
        )
        print(f"Loaded {model_name} model successfully")
    except Exception as e:
        print(f"Failed to load {model_name} model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sepsis risk for a patient"""
    try:
        data = request.json
        
        # Extract data
        patient_id = data.get('patient_id', 'unknown')
        features = np.array(data['features'])
        masks = np.array(data['masks'])
        model_name = data.get('model_name', 'grud')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Make prediction
        prediction = models[model_name].predict(features, masks)
        prediction['patient_id'] = patient_id
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'available_models': list(models.keys()),
        'model_configs': model_configs
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

if __name__ == '__main__':
    print("Starting Sepsis Prediction API...")
    print(f"Available models: {list(models.keys())}")
    app.run(host='0.0.0.0', port=5000, debug=True)
