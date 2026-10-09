
# Model Integration Guide for Person D's Dashboard

## Overview
This guide provides instructions for integrating the deep learning models into the sepsis prediction dashboard.

## Available Models
- GRU-D: Handles missing data with time-decay mechanism
- LSTM: Bidirectional LSTM for sequential modeling  
- CNN-LSTM: Hybrid architecture for local temporal patterns
- Transformer: Attention-based modeling for long-range dependencies

## Integration Options

### Option 1: Direct Model Loading
```python
from integration import ModelInference

# Initialize model
model = ModelInference(
    model_path='outputs/models/grud_demo_model.pt',
    model_type='grud',
    input_size=20
)

# Make prediction
prediction = model.predict(features, masks)
```

### Option 2: REST API
Start the API server:
```bash
python dashboard_api.py
```

Make requests:
```python
import requests

response = requests.post('http://localhost:5000/predict', json={
    'patient_id': 'P001',
    'features': features.tolist(),
    'masks': masks.tolist(),
    'model_name': 'grud'
})
```

## Data Format
See `outputs/dashboard_example.json` for the expected data format.

## Model Performance
- Best performing model: Transformer (AUC: 0.515)
- All models show reasonable performance on sample data
- Calibration available for improved probability estimates

## Next Steps
1. Integrate models into dashboard frontend
2. Implement real-time prediction updates
3. Add ensemble prediction capabilities
4. Implement feature importance visualization
