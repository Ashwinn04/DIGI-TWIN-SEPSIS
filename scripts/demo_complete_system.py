"""
Complete Sepsis Prediction System Demo
Demonstrates integration of all team members' work:
- Person A: Data preprocessing and clinical scores
- Person B: Baseline models (SIRS, qSOFA, SOFA, ML models)
- Person C: Deep learning models (GRU-D, LSTM, CNN-LSTM, Transformer)
- Person D: Explainability and dashboard
"""
import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/Users/ashwinnair/Downloads/Capstone 2'
sys.path.append(project_root)

def load_real_data():
    """Load real ICU data from Person A"""
    # Try multiple possible paths for the dataset
    dataset_paths = [
        'Dataset.csv',
        'Capstone/Dataset.csv'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"✅ Loaded real data from {path}: {df.shape[0]} records, {df.shape[1]} features")
                return df
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}, trying next path...")
                continue
    
    # If none of the paths worked
    print("❌ Dataset.csv not found. Using sample data.")
    print(f"💡 Tried paths: {', '.join(dataset_paths)}")
    return None

def create_sample_patients():
    """Create sample patients with different risk profiles"""
    patients = []
    
    # Low risk patient
    patients.append({
        'Patient_ID': 'P001',
        'HR': 75, 'O2Sat': 98, 'Temp': 36.8, 'SBP': 125, 'MAP': 85, 'DBP': 70,
        'Resp': 16, 'EtCO2': 35, 'BaseExcess': 0, 'HCO3': 24, 'FiO2': 0.21,
        'pH': 7.4, 'PaCO2': 40, 'SaO2': 98, 'AST': 25, 'BUN': 15, 'Alkalinephos': 80,
        'Calcium': 9.5, 'Chloride': 100, 'Creatinine': 0.9, 'Bilirubin_direct': 0.3,
        'Lactate': 1.0, 'Magnesium': 2.0, 'Phosphate': 3.5, 'Potassium': 4.0,
        'Bilirubin_total': 0.8, 'TroponinI': 0.01, 'Hct': 42, 'Hgb': 14,
        'PTT': 30, 'WBC': 7.5, 'Fibrinogen': 300, 'Platelets': 250,
        'Age': 45, 'Gender': 1, 'Unit1': 1, 'Unit2': 0, 'HospAdmTime': 2, 'ICULOS': 1
    })
    
    # Medium risk patient
    patients.append({
        'Patient_ID': 'P002',
        'HR': 95, 'O2Sat': 92, 'Temp': 38.2, 'SBP': 110, 'MAP': 75, 'DBP': 60,
        'Resp': 22, 'EtCO2': 30, 'BaseExcess': -2, 'HCO3': 20, 'FiO2': 0.4,
        'pH': 7.35, 'PaCO2': 45, 'SaO2': 92, 'AST': 45, 'BUN': 25, 'Alkalinephos': 120,
        'Calcium': 8.5, 'Chloride': 95, 'Creatinine': 1.5, 'Bilirubin_direct': 0.8,
        'Lactate': 2.5, 'Magnesium': 1.8, 'Phosphate': 3.0, 'Potassium': 3.5,
        'Bilirubin_total': 2.0, 'TroponinI': 0.05, 'Hct': 35, 'Hgb': 11,
        'PTT': 35, 'WBC': 12.5, 'Fibrinogen': 250, 'Platelets': 180,
        'Age': 65, 'Gender': 0, 'Unit1': 1, 'Unit2': 0, 'HospAdmTime': 4, 'ICULOS': 2
    })
    
    # High risk patient
    patients.append({
        'Patient_ID': 'P003',
        'HR': 110, 'O2Sat': 88, 'Temp': 39.1, 'SBP': 95, 'MAP': 65, 'DBP': 50,
        'Resp': 28, 'EtCO2': 25, 'BaseExcess': -5, 'HCO3': 18, 'FiO2': 0.6,
        'pH': 7.25, 'PaCO2': 50, 'SaO2': 88, 'AST': 80, 'BUN': 40, 'Alkalinephos': 200,
        'Calcium': 7.5, 'Chloride': 90, 'Creatinine': 2.5, 'Bilirubin_direct': 2.0,
        'Lactate': 4.5, 'Magnesium': 1.5, 'Phosphate': 2.5, 'Potassium': 3.0,
        'Bilirubin_total': 4.0, 'TroponinI': 0.2, 'Hct': 28, 'Hgb': 9,
        'PTT': 45, 'WBC': 18.0, 'Fibrinogen': 200, 'Platelets': 120,
        'Age': 78, 'Gender': 1, 'Unit1': 1, 'Unit2': 0, 'HospAdmTime': 6, 'ICULOS': 3
    })
    
    return patients

def calculate_clinical_scores(patient):
    """Calculate clinical scores (Person B's implementation)"""
    scores = {}
    
    # SIRS Score
    sirs_score = 0
    if patient['Temp'] > 38 or patient['Temp'] < 36:
        sirs_score += 1
    if patient['HR'] > 90:
        sirs_score += 1
    if patient['Resp'] > 20:
        sirs_score += 1
    if patient['WBC'] > 12 or patient['WBC'] < 4:
        sirs_score += 1
    scores['sirs'] = sirs_score
    
    # qSOFA Score
    qsofa_score = 0
    if patient['Resp'] >= 22:
        qsofa_score += 1
    if patient['SBP'] <= 100:
        qsofa_score += 1
    # GCS not available, so max score is 2
    scores['qsofa'] = qsofa_score
    
    # SOFA Score (partial)
    sofa_score = 0
    # Platelets
    if patient['Platelets'] < 20:
        sofa_score += 4
    elif patient['Platelets'] < 50:
        sofa_score += 3
    elif patient['Platelets'] < 100:
        sofa_score += 2
    elif patient['Platelets'] < 150:
        sofa_score += 1
    
    # Bilirubin
    if patient['Bilirubin_total'] >= 12:
        sofa_score += 4
    elif patient['Bilirubin_total'] >= 6:
        sofa_score += 3
    elif patient['Bilirubin_total'] >= 2:
        sofa_score += 2
    elif patient['Bilirubin_total'] >= 1.2:
        sofa_score += 1
    
    scores['sofa'] = sofa_score
    
    return scores

def simulate_baseline_models(patient):
    """Simulate Person B's baseline model predictions"""
    # Simple risk calculation based on clinical scores
    sirs_score = calculate_clinical_scores(patient)['sirs']
    qsofa_score = calculate_clinical_scores(patient)['qsofa']
    sofa_score = calculate_clinical_scores(patient)['sofa']
    
    # Logistic Regression (simplified)
    lr_risk = 0.1 + (sirs_score * 0.15) + (qsofa_score * 0.2) + (sofa_score * 0.1)
    lr_risk = min(lr_risk, 0.95)
    
    # Random Forest (simplified)
    rf_risk = 0.05 + (sirs_score * 0.12) + (qsofa_score * 0.18) + (sofa_score * 0.08)
    rf_risk = min(rf_risk, 0.9)
    
    # XGBoost (simplified)
    xgb_risk = 0.08 + (sirs_score * 0.14) + (qsofa_score * 0.16) + (sofa_score * 0.09)
    xgb_risk = min(xgb_risk, 0.92)
    
    return {
        'logistic_regression': {
            'risk_score': lr_risk,
            'risk_level': 'High' if lr_risk > 0.7 else 'Medium' if lr_risk > 0.3 else 'Low',
            'confidence': 0.8
        },
        'random_forest': {
            'risk_score': rf_risk,
            'risk_level': 'High' if rf_risk > 0.7 else 'Medium' if rf_risk > 0.3 else 'Low',
            'confidence': 0.85
        },
        'xgboost': {
            'risk_score': xgb_risk,
            'risk_level': 'High' if xgb_risk > 0.7 else 'Medium' if xgb_risk > 0.3 else 'Low',
            'confidence': 0.82
        }
    }

def simulate_deep_learning_models(patient):
    """Simulate Person C's deep learning model predictions"""
    # Use different features for each model to show variety
    hr_factor = patient['HR'] / 100
    temp_factor = abs(patient['Temp'] - 37) / 2
    resp_factor = patient['Resp'] / 30
    wbc_factor = patient['WBC'] / 20
    
    # GRU-D (uses temporal patterns)
    grud_risk = 0.2 + (hr_factor * 0.3) + (temp_factor * 0.2) + (resp_factor * 0.3)
    grud_risk = min(grud_risk, 0.95)
    
    # LSTM (uses sequential patterns)
    lstm_risk = 0.15 + (hr_factor * 0.25) + (temp_factor * 0.25) + (wbc_factor * 0.2)
    lstm_risk = min(lstm_risk, 0.9)
    
    # CNN-LSTM (uses spatial-temporal patterns)
    cnn_lstm_risk = 0.18 + (hr_factor * 0.28) + (temp_factor * 0.22) + (resp_factor * 0.25)
    cnn_lstm_risk = min(cnn_lstm_risk, 0.92)
    
    # Transformer (uses attention mechanisms)
    transformer_risk = 0.12 + (hr_factor * 0.32) + (temp_factor * 0.18) + (wbc_factor * 0.28)
    transformer_risk = min(transformer_risk, 0.88)
    
    return {
        'grud': {
            'risk_score': grud_risk,
            'risk_level': 'High' if grud_risk > 0.7 else 'Medium' if grud_risk > 0.3 else 'Low',
            'confidence': 0.85
        },
        'lstm': {
            'risk_score': lstm_risk,
            'risk_level': 'High' if lstm_risk > 0.7 else 'Medium' if lstm_risk > 0.3 else 'Low',
            'confidence': 0.8
        },
        'cnn_lstm': {
            'risk_score': cnn_lstm_risk,
            'risk_level': 'High' if cnn_lstm_risk > 0.7 else 'Medium' if cnn_lstm_risk > 0.3 else 'Low',
            'confidence': 0.88
        },
        'transformer': {
            'risk_score': transformer_risk,
            'risk_level': 'High' if transformer_risk > 0.7 else 'Medium' if transformer_risk > 0.3 else 'Low',
            'confidence': 0.82
        }
    }

def calculate_ensemble_prediction(all_predictions):
    """Calculate ensemble prediction from all models"""
    risk_scores = []
    for pred in all_predictions.values():
        if isinstance(pred, dict) and 'risk_score' in pred:
            risk_scores.append(pred['risk_score'])
    
    if risk_scores:
        ensemble_score = np.mean(risk_scores)
        ensemble_level = 'High' if ensemble_score > 0.7 else 'Medium' if ensemble_score > 0.3 else 'Low'
        
        # Calculate agreement score
        agreement_score = 1 - (np.std(risk_scores) / np.mean(risk_scores)) if np.mean(risk_scores) > 0 else 0
        
        return {
            'average_risk_score': float(ensemble_score),
            'risk_level': ensemble_level,
            'agreement_score': float(agreement_score),
            'recommended_action': get_recommendation(ensemble_level)
        }
    else:
        return {
            'average_risk_score': 0.5,
            'risk_level': 'Medium',
            'agreement_score': 0.0,
            'recommended_action': 'Continue monitoring'
        }

def get_recommendation(risk_level):
    """Get clinical recommendation based on risk level"""
    if risk_level == 'High':
        return 'Immediate clinical attention required - Consider sepsis protocol activation'
    elif risk_level == 'Medium':
        return 'Close monitoring recommended - Review patient status and consider additional assessments'
    else:
        return 'Continue routine monitoring - Patient appears stable'

def generate_feature_importance(patient):
    """Generate feature importance for explainability (Person D's work)"""
    # Calculate importance based on deviation from normal values
    normal_values = {
        'HR': 80, 'Temp': 37, 'Resp': 16, 'WBC': 8, 'SBP': 120,
        'O2Sat': 95, 'Creatinine': 1.0, 'Bilirubin_total': 1.0, 'Lactate': 1.0
    }
    
    importance = []
    for feature, normal_val in normal_values.items():
        if feature in patient:
            value = patient[feature]
            deviation = abs(value - normal_val) / normal_val
            importance.append({
                'feature': feature.lower(),
                'importance': min(deviation, 1.0),
                'trend': 'increasing' if value > normal_val else 'stable',
                'current_value': value,
                'normal_range': f"{normal_val}±{normal_val*0.1:.1f}"
            })
    
    # Sort by importance
    importance.sort(key=lambda x: x['importance'], reverse=True)
    return {'top_features': importance[:5]}

def generate_risk_trajectory(patient_id, current_risk):
    """Generate risk trajectory over time"""
    # Generate different trajectories based on patient risk
    if current_risk > 0.7:
        # High risk - increasing trend
        base_risk = 0.3
        trend = 0.05
    elif current_risk > 0.3:
        # Medium risk - stable with some variation
        base_risk = 0.4
        trend = 0.01
    else:
        # Low risk - decreasing trend
        base_risk = 0.6
        trend = -0.02
    
    trajectory = []
    for i in range(24):
        risk = base_risk + (trend * i) + np.random.normal(0, 0.05)
        risk = np.clip(risk, 0, 1)
        trajectory.append({
            'timestamp': (datetime.now() - timedelta(hours=23-i)).isoformat(),
            'risk_score': risk
        })
    
    return trajectory

def run_complete_demo():
    """Run the complete system demo"""
    print("🏥 Sepsis Prediction System - Complete Demo")
    print("=" * 50)
    
    # Load data
    df = load_real_data()
    
    # Create sample patients
    patients = create_sample_patients()
    
    print(f"\n📊 Processing {len(patients)} patients...")
    
    results = []
    
    for patient in patients:
        print(f"\n👤 Processing Patient {patient['Patient_ID']}...")
        
        # Calculate clinical scores (Person B's work)
        clinical_scores = calculate_clinical_scores(patient)
        print(f"   📋 Clinical Scores: SIRS={clinical_scores['sirs']}, qSOFA={clinical_scores['qsofa']}, SOFA={clinical_scores['sofa']}")
        
        # Get baseline model predictions (Person B's work)
        baseline_preds = simulate_baseline_models(patient)
        print(f"   🤖 Baseline Models: LR={baseline_preds['logistic_regression']['risk_score']:.3f}, RF={baseline_preds['random_forest']['risk_score']:.3f}, XGB={baseline_preds['xgboost']['risk_score']:.3f}")
        
        # Get deep learning predictions (Person C's work)
        dl_preds = simulate_deep_learning_models(patient)
        print(f"   🧠 Deep Learning: GRU-D={dl_preds['grud']['risk_score']:.3f}, LSTM={dl_preds['lstm']['risk_score']:.3f}, CNN-LSTM={dl_preds['cnn_lstm']['risk_score']:.3f}, Transformer={dl_preds['transformer']['risk_score']:.3f}")
        
        # Combine all predictions
        all_predictions = {**baseline_preds, **dl_preds}
        
        # Calculate ensemble prediction
        ensemble_pred = calculate_ensemble_prediction(all_predictions)
        print(f"   🎯 Ensemble Prediction: {ensemble_pred['average_risk_score']:.3f} ({ensemble_pred['risk_level']})")
        
        # Generate feature importance (Person D's work)
        feature_importance = generate_feature_importance(patient)
        print(f"   🔍 Top Feature: {feature_importance['top_features'][0]['feature']} (importance: {feature_importance['top_features'][0]['importance']:.3f})")
        
        # Generate risk trajectory
        risk_trajectory = generate_risk_trajectory(patient['Patient_ID'], ensemble_pred['average_risk_score'])
        
        # Store results
        result = {
            'patient_id': patient['Patient_ID'],
            'patient_data': patient,
            'clinical_scores': clinical_scores,
            'baseline_predictions': baseline_preds,
            'deep_learning_predictions': dl_preds,
            'ensemble_prediction': ensemble_pred,
            'feature_importance': feature_importance,
            'risk_trajectory': risk_trajectory,
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
    
    # Save results
    with open('outputs/complete_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Demo completed! Results saved to outputs/complete_demo_results.json")
    
    # Summary
    print(f"\n📈 Summary:")
    for result in results:
        pred = result['ensemble_prediction']
        print(f"   Patient {result['patient_id']}: {pred['risk_level']} risk ({pred['average_risk_score']:.3f}) - {pred['recommended_action']}")
    
    return results

if __name__ == "__main__":
    results = run_complete_demo()
