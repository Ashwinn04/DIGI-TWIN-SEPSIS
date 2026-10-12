"""
FastAPI microservice for sepsis prediction
Hospital-ready API for integration with clinical systems
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sepsis Prediction API",
    description="Clinical AI API for early sepsis detection and risk assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VitalSigns(BaseModel):
    heart_rate: float = Field(..., ge=30, le=200, description="Heart rate in bpm")
    map: float = Field(..., ge=40, le=150, description="Mean arterial pressure in mm Hg")
    temperature: float = Field(..., ge=30.0, le=45.0, description="Temperature in °C")
    respiratory_rate: float = Field(..., ge=5, le=50, description="Respiratory rate in breaths/min")

class LabValues(BaseModel):
    lactate: float = Field(..., ge=0.1, le=20.0, description="Lactate in mmol/L")
    wbc: float = Field(..., ge=0.1, le=50.0, description="White blood cell count in ×10³/μL")
    creatinine: float = Field(..., ge=0.1, le=10.0, description="Creatinine in mg/dL")
    platelets: Optional[float] = Field(None, ge=10, le=1000, description="Platelet count in ×10³/μL")
    bilirubin: Optional[float] = Field(None, ge=0.1, le=50.0, description="Total bilirubin in mg/dL")

class Demographics(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Patient gender")
    admission_type: Optional[str] = Field(None, description="Type of admission")

class PatientData(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    vital_signs: VitalSigns
    lab_values: LabValues
    demographics: Demographics
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional clinical data")

class PredictionResponse(BaseModel):
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Sepsis risk score")
    risk_level: str = Field(..., description="Risk level: low, medium, or high")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    top_features: List[Dict[str, Any]] = Field(..., description="Top contributing features")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    clinical_scores: Dict[str, float] = Field(..., description="Clinical scores (SIRS, qSOFA, SOFA)")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_versions: Dict[str, str] = Field(..., description="Model versions used")

class ExplanationResponse(BaseModel):
    patient_id: str
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP feature importance")
    lime_explanation: Optional[List[Dict[str, Any]]] = Field(None, description="LIME explanation")
    integrated_gradients: Optional[Dict[str, float]] = Field(None, description="Integrated gradients")
    case_narrative: str = Field(..., description="Human-readable case summary")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="Confidence intervals")

class ModelStatus(BaseModel):
    models_loaded: bool
    model_count: int
    last_updated: datetime
    model_versions: Dict[str, str]
    system_status: str

class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]
    include_explanations: bool = Field(False, description="Include explainability analysis")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    processing_time: float
    success_count: int
    error_count: int

# Global variables for model caching
model_cache = {}
executor = ThreadPoolExecutor(max_workers=4)

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "service": "sepsis-prediction-api"
    }

# Model status endpoint
@app.get("/models/status", response_model=ModelStatus)
async def get_model_status():
    """Get status of the models that are actually loaded"""
    try:
        system = _get_system()
        names = sorted(set(system.baseline_models) | set(system.deep_learning_models))
        return ModelStatus(
            models_loaded=True,
            model_count=len(names),
            last_updated=datetime.now(),
            model_versions={n: system.model_set for n in names},
            system_status="operational"
        )
    except Exception as e:
        return ModelStatus(models_loaded=False, model_count=0, last_updated=datetime.now(),
                           model_versions={}, system_status=f"unavailable: {e}")

# Single patient prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(patient_data: PatientData, background_tasks: BackgroundTasks):
    """Predict sepsis risk for a single patient"""
    try:
        logger.info(f"Processing prediction request for patient {patient_data.patient_id}")
        
        # Convert to dictionary for processing
        patient_dict = patient_data.dict()
        
        # Run prediction in thread pool
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(
            executor, 
            _predict_single_patient, 
            patient_dict
        )
        
        # Log prediction for audit
        background_tasks.add_task(log_prediction, patient_data.patient_id, prediction)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction failed for patient {patient_data.patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_sepsis_batch(request: BatchPredictionRequest):
    """Predict sepsis risk for multiple patients"""
    try:
        logger.info(f"Processing batch prediction for {len(request.patients)} patients")
        
        start_time = datetime.now()
        predictions = []
        success_count = 0
        error_count = 0
        
        # Process each patient
        for patient_data in request.patients:
            try:
                patient_dict = patient_data.dict()
                prediction = _predict_single_patient(patient_dict)
                predictions.append(PredictionResponse(**prediction))
                success_count += 1
            except Exception as e:
                logger.error(f"Batch prediction failed for patient {patient_data.patient_id}: {str(e)}")
                error_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processing_time=processing_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Explainability endpoint
@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(patient_data: PatientData):
    """Get explainability for a prediction"""
    try:
        logger.info(f"Generating explanation for patient {patient_data.patient_id}")
        
        # Convert to dictionary
        patient_dict = patient_data.dict()
        
        # Generate explanation
        explanation = _generate_explanation(patient_dict)
        
        return ExplanationResponse(**explanation)
        
    except Exception as e:
        logger.error(f"Explanation failed for patient {patient_data.patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# Clinical scores endpoint
@app.post("/clinical-scores")
async def calculate_clinical_scores(patient_data: PatientData):
    """Calculate clinical scores (SIRS, qSOFA, SOFA)"""
    try:
        patient_dict = patient_data.dict()
        
        # Calculate clinical scores
        sirs_score = _calculate_sirs_score(patient_dict)
        qsofa_score = _calculate_qsofa_score(patient_dict)
        sofa_score = _calculate_sofa_score(patient_dict)
        
        return {
            "patient_id": patient_data.patient_id,
            "sirs_score": sirs_score,
            "qsofa_score": qsofa_score,
            "sofa_score": sofa_score,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Clinical scores calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clinical scores calculation failed: {str(e)}")

# Model comparison endpoint
@app.post("/compare")
async def compare_models(patient_data: PatientData):
    """Compare predictions across all models"""
    try:
        patient_dict = patient_data.dict()
        
        system = _get_system()
        pred = system.get_comprehensive_prediction(_to_record(patient_dict))
        if pred is None:
            raise RuntimeError("Model system returned no prediction")
        model_predictions = {
            _MODEL_NAMES.get(n, n): {"risk_score": float(p["risk_score"]), "confidence": float(p.get("confidence", 0.0)),
                                     "risk_level": _model_level(p["risk_level"])}
            for n, p in pred["individual_predictions"].items() if isinstance(p, dict) and "risk_score" in p}

        return {
            "patient_id": patient_data.patient_id,
            "model_set": system.model_set,
            "model_predictions": model_predictions,
            "ensemble_prediction": _calculate_ensemble_prediction(model_predictions, pred["ensemble_prediction"]),
            "model_agreement": _calculate_model_agreement(model_predictions),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

# Helper functions
# ---------------------------------------------------------------------------------------------
# Model access. Everything below uses the same system as the dashboard and CLI (integration_real);
# SEPSIS_MODEL_SET=protocol serves the models evaluated in outputs/results/final_results.csv.
# ---------------------------------------------------------------------------------------------
_system = None
_API_NAMES = {"HR": "heart_rate", "Temp": "temperature", "Resp": "respiratory_rate", "MAP": "map", "Lactate": "lactate",
              "WBC": "wbc", "Creatinine": "creatinine", "Platelets": "platelets", "Bilirubin_total": "bilirubin"}
_MODEL_NAMES = {"grud": "gru_d"}


def _get_system():
    global _system
    if _system is None:
        from integration_real import get_integration_system
        _system = get_integration_system()
    if _system is None or not getattr(_system, "is_initialized", False):
        raise RuntimeError("Prediction system is not initialised")
    return _system


def _to_record(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the API schema into the feature names the models were trained on."""
    v, l, d = patient_dict["vital_signs"], patient_dict["lab_values"], patient_dict["demographics"]
    rec = {"HR": v["heart_rate"], "MAP": v["map"], "Temp": v["temperature"], "Resp": v["respiratory_rate"],
           "Lactate": l["lactate"], "WBC": l["wbc"], "Creatinine": l["creatinine"],
           "Age": d["age"], "Gender": d["gender"]}
    if l.get("platelets") is not None:
        rec["Platelets"] = l["platelets"]
    if l.get("bilirubin") is not None:
        rec["Bilirubin_total"] = l["bilirubin"]
    rec.update(patient_dict.get("additional_data") or {})
    return rec


def _model_level(level: str) -> str:
    return str(level).lower()


def _predict_single_patient(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict sepsis risk for a single patient with the project's models (no simulation)."""
    system = _get_system()
    record = _to_record(patient_dict)
    pred = system.get_comprehensive_prediction(record)
    if pred is None:
        raise RuntimeError("Model system returned no prediction")
    ens = pred["ensemble_prediction"]
    risk_score = float(np.clip(ens["average_risk_score"], 0.0, 1.0))
    risk_level = _model_level(ens["risk_level"])
    confidence = float(np.clip(ens.get("agreement_score", 0.0), 0.0, 1.0))   # agreement among models

    # Reference-range deviations (a descriptive heuristic, not a model attribution)
    top_features = []
    for item in system.get_feature_importance(record).get("top_features", []):
        canon = next((k for k in _API_NAMES if k.lower() == item["feature"].lower()), None)
        top_features.append({"feature": _API_NAMES.get(canon, item["feature"]), "importance": item["importance"],
                             "value": record.get(canon, patient_dict["vital_signs"].get(_API_NAMES.get(canon, ""), 0.0)),
                             "method": "deviation from reference range"})

    # Generate recommendations
    recommendations = _generate_recommendations(risk_level, top_features)
    
    # Calculate clinical scores
    clinical_scores = {
        "sirs": _calculate_sirs_score(patient_dict),
        "qsofa": _calculate_qsofa_score(patient_dict),
        "sofa": _calculate_sofa_score(patient_dict)
    }
    
    return {
        "patient_id": patient_dict["patient_id"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "top_features": top_features,
        "recommendations": recommendations,
        "clinical_scores": clinical_scores,
        "timestamp": datetime.now(),
        "model_versions": {
            "model_set": system.model_set,
            "models": ", ".join(sorted(n for n in pred["individual_predictions"] if n != "clinical_scores")),
        }
    }

def _generate_explanation(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Descriptive explanation. SHAP, LIME and integrated gradients are NOT computed by this service, so those
    fields are null; the narrative lists deviations from reference ranges next to the model's own score."""
    pred = _predict_single_patient(patient_dict)
    parts = [f"{f['feature']} {f['value']:.1f}" for f in pred["top_features"][:3]]
    narrative = (f"Patient {patient_dict['patient_id']}: model set '{pred['model_versions']['model_set']}' gives "
                 f"{pred['risk_score']:.3f} ({pred['risk_level']} risk). Largest deviations from reference range: "
                 f"{', '.join(parts) if parts else 'none'}. This is a descriptive summary, not a causal or "
                 f"model-attribution explanation.")
    return {
        "patient_id": patient_dict["patient_id"],
        "shap_values": None,
        "lime_explanation": None,
        "integrated_gradients": None,
        "case_narrative": narrative,
        "confidence_intervals": {}
    }

def _calculate_sirs_score(patient_dict: Dict[str, Any]) -> int:
    """Calculate SIRS score"""
    score = 0
    vital_signs = patient_dict["vital_signs"]
    lab_values = patient_dict["lab_values"]
    
    # Temperature
    if vital_signs["temperature"] > 38.0 or vital_signs["temperature"] < 36.0:
        score += 1
    
    # Heart rate
    if vital_signs["heart_rate"] > 90:
        score += 1
    
    # Respiratory rate
    if vital_signs["respiratory_rate"] > 20:
        score += 1
    
    # WBC
    if lab_values["wbc"] > 12 or lab_values["wbc"] < 4:
        score += 1
    
    return score

def _calculate_qsofa_score(patient_dict: Dict[str, Any]) -> int:
    """Calculate qSOFA score"""
    score = 0
    vital_signs = patient_dict["vital_signs"]
    
    # Respiratory rate
    if vital_signs["respiratory_rate"] >= 22:
        score += 1
    
    # Systolic blood pressure (approximated from MAP)
    sbp = vital_signs["map"] * 1.3  # Rough approximation
    if sbp <= 100:
        score += 1
    
    # GCS not available, so max score is 2
    return score

def _calculate_sofa_score(patient_dict: Dict[str, Any]) -> int:
    """Calculate partial SOFA score"""
    score = 0
    lab_values = patient_dict["lab_values"]
    
    # Platelets
    if lab_values.get("platelets"):
        platelets = lab_values["platelets"]
        if platelets < 20:
            score += 4
        elif platelets < 50:
            score += 3
        elif platelets < 100:
            score += 2
        elif platelets < 150:
            score += 1
    
    # Bilirubin
    if lab_values.get("bilirubin"):
        bilirubin = lab_values["bilirubin"]
        if bilirubin >= 12:
            score += 4
        elif bilirubin >= 6:
            score += 3
        elif bilirubin >= 2:
            score += 2
        elif bilirubin >= 1.2:
            score += 1
    
    return score

def _generate_recommendations(risk_level: str, top_features: List[Dict[str, Any]]) -> List[str]:
    """Generate clinical recommendations based on risk level and features"""
    recommendations = []
    
    if risk_level == "high":
        recommendations.extend([
            "🚨 Immediate intervention required",
            "Consider sepsis protocol activation",
            "Monitor vital signs every 15 minutes",
            "Prepare for potential ICU transfer"
        ])
    elif risk_level == "medium":
        recommendations.extend([
            "⚠️ Close monitoring recommended",
            "Repeat laboratory tests in 2-4 hours",
            "Monitor vital signs every 30 minutes",
            "Consider early warning system activation"
        ])
    else:
        recommendations.extend([
            "✅ Continue routine monitoring",
            "Standard vital sign monitoring",
            "Regular laboratory follow-up"
        ])
    
    # Add feature-specific recommendations
    for feature in top_features:
        if feature["feature"] == "lactate" and feature["value"] > 2.0:
            recommendations.append("High lactate - consider lactate clearance monitoring")
        elif feature["feature"] == "heart_rate" and feature["value"] > 100:
            recommendations.append("Elevated heart rate - monitor for tachycardia")
    
    return recommendations

def _get_all_model_predictions(patient_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per-model predictions from the loaded system."""
    system = _get_system()
    pred = system.get_comprehensive_prediction(_to_record(patient_dict))
    if pred is None:
        raise RuntimeError("Model system returned no prediction")
    out = {}
    for name, p in pred["individual_predictions"].items():
        if isinstance(p, dict) and "risk_score" in p:
            out[_MODEL_NAMES.get(name, name)] = {"risk_score": float(p["risk_score"]),
                                                 "confidence": float(p.get("confidence", 0.0)),
                                                 "risk_level": _model_level(p["risk_level"])}
    return out

def _calculate_ensemble_prediction(model_predictions: Dict[str, Dict[str, Any]],
                                   system_ensemble: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensemble summary. When the system's own ensemble is supplied its calibrated score and level are used."""
    confidences = [p["confidence"] for p in model_predictions.values()]
    if system_ensemble is not None:
        return {"risk_score": float(system_ensemble["average_risk_score"]),
                "confidence": float(np.clip(system_ensemble.get("agreement_score", 0.0), 0.0, 1.0)),
                "risk_level": _model_level(system_ensemble["risk_level"])}
    scores = [p["risk_score"] for p in model_predictions.values()]
    return {"risk_score": float(np.mean(scores)), "confidence": float(np.mean(confidences)) if confidences else 0.0,
            "risk_level": "high" if np.mean(scores) >= 0.7 else "medium" if np.mean(scores) >= 0.4 else "low"}

def _calculate_model_agreement(model_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate model agreement metrics"""
    scores = [pred["risk_score"] for pred in model_predictions.values()]
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "agreement_score": 1.0 - np.std(scores),  # Lower std = higher agreement
        "consensus_level": "high" if np.std(scores) < 0.1 else "medium" if np.std(scores) < 0.2 else "low"
    }

def log_prediction(patient_id: str, prediction: Dict[str, Any]):
    """Log prediction for audit trail"""
    log_entry = {
        "patient_id": patient_id,
        "timestamp": datetime.now(),
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "confidence": prediction["confidence"]
    }
    
    # Save to log file
    log_file = "outputs/prediction_log.jsonl"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
