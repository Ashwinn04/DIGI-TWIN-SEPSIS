#!/usr/bin/env python3
"""
Command-line tool for predicting sepsis risk from new patient data
Usage:
    python predict_new_data.py --json patient_data.json
    python predict_new_data.py --csv patient_data.csv
    python predict_new_data.py --interactive
    python predict_new_data.py --data "HR=95,Temp=38.2,Resp=22,WBC=12.5,Lactate=2.5,Age=65"
"""
import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integration_real import get_integration_system


def parse_data_string(data_string: str) -> Dict[str, Any]:
    """Parse comma-separated key=value pairs into a dictionary"""
    patient_data = {}
    for pair in data_string.split(','):
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    patient_data[key] = float(value)
                else:
                    patient_data[key] = int(value)
            except ValueError:
                patient_data[key] = value
    return patient_data


def load_patient_data_from_json(json_path: str) -> Dict[str, Any]:
    """Load patient data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Handle nested structure (vital_signs, lab_values, etc.)
    if 'vital_signs' in data or 'lab_values' in data:
        # Flatten the structure
        patient_data = {}
        if 'patient_id' in data:
            patient_data['Patient_ID'] = data['patient_id']
        if 'demographics' in data:
            demos = data['demographics']
            if 'age' in demos:
                patient_data['Age'] = demos['age']
            if 'gender' in demos:
                patient_data['Gender'] = demos['gender']
        if 'vital_signs' in data:
            vs = data['vital_signs']
            patient_data['HR'] = vs.get('heart_rate', vs.get('HR', 80))
            patient_data['SBP'] = vs.get('sbp', vs.get('SBP', 120))
            patient_data['DBP'] = vs.get('dbp', vs.get('DBP', 80))
            patient_data['MAP'] = vs.get('map', vs.get('MAP', 75))
            patient_data['Temp'] = vs.get('temperature', vs.get('Temp', 37))
            patient_data['Resp'] = vs.get('respiratory_rate', vs.get('Resp', 16))
            patient_data['O2Sat'] = vs.get('oxygen_saturation', vs.get('O2Sat', 95))
        if 'lab_values' in data:
            labs = data['lab_values']
            patient_data['Lactate'] = labs.get('lactate', labs.get('Lactate', 1.5))
            patient_data['WBC'] = labs.get('wbc', labs.get('WBC', 8))
            patient_data['Creatinine'] = labs.get('creatinine', labs.get('Creatinine', 1.0))
            patient_data['Platelets'] = labs.get('platelets', labs.get('Platelets', 250))
            patient_data['Bilirubin_total'] = labs.get('bilirubin', labs.get('Bilirubin_total', 1.0))
        return patient_data
    return data


def load_patient_data_from_csv(csv_path: str) -> Dict[str, Any]:
    """Load patient data from CSV file (first row)"""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for CSV file support. Install with: pip install pandas")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("CSV file is empty")
    # Use first row
    patient_data = df.iloc[0].to_dict()
    return patient_data


def interactive_input() -> Dict[str, Any]:
    """Interactive input for patient data"""
    print("\n" + "="*60)
    print("📋 Enter Patient Data (press Enter to use defaults)")
    print("="*60)
    
    patient_data = {}
    
    # Demographics
    print("\n👤 Demographics:")
    patient_data['Patient_ID'] = input("Patient ID: ").strip() or "P001"
    age = input("Age (default: 65): ").strip()
    patient_data['Age'] = int(age) if age else 65
    gender = input("Gender (Male/Female, default: Male): ").strip()
    patient_data['Gender'] = 1 if gender.lower() in ['male', 'm', ''] else 0
    
    # Vital Signs
    print("\n💓 Vital Signs:")
    hr = input("Heart Rate (bpm, default: 80): ").strip()
    patient_data['HR'] = float(hr) if hr else 80.0
    sbp = input("Systolic BP (mm Hg, default: 120): ").strip()
    patient_data['SBP'] = float(sbp) if sbp else 120.0
    dbp = input("Diastolic BP (mm Hg, default: 80): ").strip()
    patient_data['DBP'] = float(dbp) if dbp else 80.0
    map_val = input("MAP (mm Hg, default: 75): ").strip()
    patient_data['MAP'] = float(map_val) if map_val else 75.0
    temp = input("Temperature (°C, default: 37.0): ").strip()
    patient_data['Temp'] = float(temp) if temp else 37.0
    resp = input("Respiratory Rate (/min, default: 16): ").strip()
    patient_data['Resp'] = float(resp) if resp else 16.0
    o2sat = input("O2 Saturation (%, default: 95): ").strip()
    patient_data['O2Sat'] = float(o2sat) if o2sat else 95.0
    
    # Lab Values
    print("\n🧪 Laboratory Values:")
    lactate = input("Lactate (mmol/L, default: 1.5): ").strip()
    patient_data['Lactate'] = float(lactate) if lactate else 1.5
    wbc = input("WBC (×10³/μL, default: 8.0): ").strip()
    patient_data['WBC'] = float(wbc) if wbc else 8.0
    creatinine = input("Creatinine (mg/dL, default: 1.0): ").strip()
    patient_data['Creatinine'] = float(creatinine) if creatinine else 1.0
    platelets = input("Platelets (×10³/μL, default: 250): ").strip()
    patient_data['Platelets'] = float(platelets) if platelets else 250.0
    bilirubin = input("Bilirubin Total (mg/dL, default: 1.0): ").strip()
    patient_data['Bilirubin_total'] = float(bilirubin) if bilirubin else 1.0
    
    # Optional fields
    print("\n📊 Optional Fields (press Enter to skip):")
    fio2 = input("FiO2 (%, default: 21): ").strip()
    if fio2:
        patient_data['FiO2'] = float(fio2)
    ph = input("pH (default: 7.4): ").strip()
    if ph:
        patient_data['pH'] = float(ph)
    paco2 = input("PaCO2 (mmHg, default: 40): ").strip()
    if paco2:
        patient_data['PaCO2'] = float(paco2)
    
    return patient_data


def format_risk_output(prediction_result: Dict[str, Any], patient_id: str = "Unknown") -> str:
    """Format prediction results for terminal output"""
    if prediction_result is None:
        return "❌ Error: Could not generate prediction"
    
    output = []
    output.append("\n" + "="*70)
    output.append("🏥 SEPSIS RISK PREDICTION RESULTS")
    output.append("="*70)
    output.append(f"Patient ID: {patient_id}")
    output.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    
    # Ensemble Prediction
    ensemble = prediction_result.get('ensemble_prediction', {})
    risk_score = ensemble.get('average_risk_score', 0.0)
    risk_level = ensemble.get('risk_level', 'Unknown')
    recommended_action = ensemble.get('recommended_action', 'N/A')
    
    output.append("🎯 ENSEMBLE PREDICTION")
    output.append("-"*70)
    
    # Risk score with visual indicator
    risk_bar_length = int(risk_score * 50)
    risk_bar = "█" * risk_bar_length + "░" * (50 - risk_bar_length)
    
    if risk_level == 'High':
        risk_indicator = "🔴 HIGH RISK"
    elif risk_level == 'Medium':
        risk_indicator = "🟡 MEDIUM RISK"
    else:
        risk_indicator = "🟢 LOW RISK"
    
    output.append(f"Risk Level: {risk_indicator}")
    output.append(f"Risk Score: {risk_score:.4f} ({risk_score*100:.2f}%)")
    output.append(f"Risk Bar:  [{risk_bar}]")
    output.append(f"Recommendation: {recommended_action}")
    output.append("")
    
    # Individual Model Predictions
    individual = prediction_result.get('individual_predictions', {})
    if individual:
        output.append("🤖 INDIVIDUAL MODEL PREDICTIONS")
        output.append("-"*70)
        
        # Baseline models
        baseline_models = ['logistic_regression', 'random_forest', 'xgboost']
        for model_name in baseline_models:
            if model_name in individual:
                pred = individual[model_name]
                if isinstance(pred, dict) and 'risk_score' in pred:
                    score = pred.get('risk_score', 0.0)
                    level = pred.get('risk_level', 'Unknown')
                    conf = pred.get('confidence', 0.0)
                    output.append(f"  {model_name.replace('_', ' ').title():25s} | "
                                f"Score: {score:.4f} | Level: {level:6s} | Confidence: {conf:.2f}")
        
        # Deep learning models
        dl_models = ['grud', 'lstm', 'cnn_lstm', 'transformer']
        for model_name in dl_models:
            if model_name in individual:
                pred = individual[model_name]
                if isinstance(pred, dict) and 'risk_score' in pred:
                    score = pred.get('risk_score', 0.0)
                    level = pred.get('risk_level', 'Unknown')
                    conf = pred.get('confidence', 0.0)
                    output.append(f"  {model_name.upper():25s} | "
                                f"Score: {score:.4f} | Level: {level:6s} | Confidence: {conf:.2f}")
        
        output.append("")
    
    # Clinical Scores
    if 'clinical_scores' in individual:
        clinical = individual['clinical_scores']
        output.append("📋 CLINICAL SCORES")
        output.append("-"*70)
        output.append(f"  SIRS Score:  {clinical.get('sirs', 0)}/4")
        output.append(f"  qSOFA Score:  {clinical.get('qsofa', 0)}/3")
        output.append(f"  SOFA Score:  {clinical.get('sofa', 0)}/24 (partial)")
        output.append("")
    
    # Thresholds and Flags
    output.append("⚙️  RISK THRESHOLDS")
    output.append("-"*70)
    op = ensemble.get('operating_points')
    if op:
        output.append(f"  High Risk:   calibrated score ≥ {op['high_threshold']:.3f}  (validation specificity 95%)")
        output.append(f"  Medium Risk: calibrated score ≥ {op['medium_threshold']:.3f}  (validation specificity 80%)")
        output.append(f"  Low Risk:    below {op['medium_threshold']:.3f}")
        output.append(f"  Scores are calibrated probabilities of onset within {op.get('horizon_h', 6)} h (base rate about 1%).")
    else:
        output.append(f"  High Risk Threshold:   ≥ 0.70 (70%)")
        output.append(f"  Medium Risk Threshold: ≥ 0.30 (30%)")
        output.append(f"  Low Risk Threshold:    < 0.30 (30%)")
    output.append("")
    
    if risk_level == 'High':
        output.append("🚨 FLAG: HIGH RISK - Immediate clinical attention required!")
    elif risk_level == 'Medium':
        output.append("⚠️  FLAG: MEDIUM RISK - Close monitoring recommended")
    else:
        output.append("✅ FLAG: LOW RISK - Continue routine monitoring")
    
    output.append("="*70)
    output.append("")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Predict sepsis risk from new patient data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON file
  python predict_new_data.py --json patient_data.json
  
  # From CSV file
  python predict_new_data.py --csv patient_data.csv
  
  # Interactive input
  python predict_new_data.py --interactive
  
  # From command-line data string
  python predict_new_data.py --data "HR=95,Temp=38.2,Resp=22,WBC=12.5,Lactate=2.5,Age=65"
        """
    )
    
    parser.add_argument('--json', type=str, help='Path to JSON file with patient data')
    parser.add_argument('--csv', type=str, help='Path to CSV file with patient data')
    parser.add_argument('--interactive', action='store_true', help='Interactive data entry')
    parser.add_argument('--data', type=str, help='Comma-separated key=value pairs (e.g., "HR=95,Temp=38.2")')
    parser.add_argument('--output', type=str, help='Output file path (optional, saves results to file)')
    
    args = parser.parse_args()
    
    # Load integration system
    print("🔄 Loading sepsis prediction system...")
    system = get_integration_system()
    
    if not system.is_initialized:
        print("❌ Error: System failed to initialize")
        sys.exit(1)
    
    print("✅ System loaded successfully!")
    print(f"   Baseline models: {len(system.baseline_models)}")
    print(f"   Deep learning models: {len(system.deep_learning_models)}")
    print("")
    
    # Load patient data
    patient_data = None
    
    if args.json:
        if not os.path.exists(args.json):
            print(f"❌ Error: JSON file not found: {args.json}")
            sys.exit(1)
        print(f"📄 Loading patient data from JSON: {args.json}")
        patient_data = load_patient_data_from_json(args.json)
    elif args.csv:
        if not os.path.exists(args.csv):
            print(f"❌ Error: CSV file not found: {args.csv}")
            sys.exit(1)
        print(f"📄 Loading patient data from CSV: {args.csv}")
        patient_data = load_patient_data_from_csv(args.csv)
    elif args.interactive:
        patient_data = interactive_input()
    elif args.data:
        print(f"📝 Parsing patient data from command line...")
        patient_data = parse_data_string(args.data)
    else:
        parser.print_help()
        print("\n❌ Error: Please provide one of --json, --csv, --interactive, or --data")
        sys.exit(1)
    
    if not patient_data:
        print("❌ Error: No patient data loaded")
        sys.exit(1)
    
    # Get patient ID
    patient_id = patient_data.get('Patient_ID', patient_data.get('patient_id', 'Unknown'))
    
    # Make prediction
    print(f"\n🔍 Generating prediction for patient {patient_id}...")
    print("")
    
    try:
        prediction_result = system.get_comprehensive_prediction(patient_data)
        
        # Format and display output
        output_text = format_risk_output(prediction_result, patient_id)
        print(output_text)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
                f.write("\n\n# Raw JSON Output\n")
                f.write(json.dumps(prediction_result, indent=2, default=str))
            print(f"💾 Results saved to: {args.output}")
        
        # Also save raw JSON for programmatic use
        json_output_path = f"outputs/prediction_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('outputs', exist_ok=True)
        with open(json_output_path, 'w') as f:
            json.dump(prediction_result, f, indent=2, default=str)
        print(f"💾 Raw JSON saved to: {json_output_path}")
        
    except Exception as e:
        print(f"❌ Error generating prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

