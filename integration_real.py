"""
Real Integration with Person B's Baseline Models and Person C's Deep Learning Models
This module provides seamless integration between all team members' work
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# torch and xgboost each bundle libomp on macOS: with multiple OpenMP threads, unpickling xgboost after
# torch segfaults and torch ops after xgboost deadlock. One OpenMP thread avoids both (set before import).
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import xgboost  # noqa: F401
except ImportError:
    pass
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')


def _warn_if_partial_load(label: str, model_state: dict, compatible: dict) -> None:
    """A checkpoint that only partly fits the model leaves the rest randomly initialised."""
    if len(compatible) < len(model_state):
        missing = sorted(set(model_state) - set(compatible))
        print(f"  ⚠️ {label}: only {len(compatible)}/{len(model_state)} tensors loaded from checkpoint "
              f"(random init for e.g. {missing[:3]}); predictions from this model are unreliable")


class TorchModelEnsemble:
    """Utility wrapper for averaging predictions across multiple PyTorch checkpoints."""

    def __init__(self, name: str, models: List[torch.nn.Module], expected_input_size: int, expected_seq_len: int = 24):
        self.name = name
        self.models = models
        self.expected_input_size = expected_input_size
        self.expected_seq_len = expected_seq_len
        for model in self.models:
            model.eval()
            model.to('cpu')

    def predict(self, features: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Generate an averaged risk score across ensemble members."""

        if features.ndim == 2:
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif features.ndim == 3:
            features_tensor = torch.tensor(features, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected feature shape for {self.name}: {features.shape}")

        batch_size, seq_len, num_features = features_tensor.shape

        # Pad or truncate features to match expected input size
        if num_features != self.expected_input_size:
            if num_features < self.expected_input_size:
                # Pad with zeros
                pad_size = self.expected_input_size - num_features
                padding = torch.zeros((batch_size, seq_len, pad_size), dtype=torch.float32)
                features_tensor = torch.cat([features_tensor, padding], dim=2)
            else:
                # Truncate to expected size
                features_tensor = features_tensor[:, :, :self.expected_input_size]
            num_features = self.expected_input_size

        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.to(features_tensor.dtype)
            
            # Pad or truncate mask to match features
            if mask_tensor.shape[2] != num_features:
                if mask_tensor.shape[2] < num_features:
                    pad_size = num_features - mask_tensor.shape[2]
                    padding = torch.zeros((mask_tensor.shape[0], mask_tensor.shape[1], pad_size), dtype=torch.float32)
                    mask_tensor = torch.cat([mask_tensor, padding], dim=2)
                else:
                    mask_tensor = mask_tensor[:, :, :num_features]
        else:
            mask_tensor = torch.ones((batch_size, seq_len, num_features), dtype=torch.float32)

        if mask_tensor.shape != features_tensor.shape:
            mask_tensor = torch.ones_like(features_tensor)

        # Ensure sequence length similar to training distribution for DL models
        min_seq_len = max(4, int(getattr(self, 'expected_seq_len', 24) or 24))
        if features_tensor.shape[1] < min_seq_len:
            pad_length = min_seq_len - features_tensor.shape[1]
            pad_features = features_tensor[:, -1:, :].repeat(1, pad_length, 1)
            pad_mask = mask_tensor[:, -1:, :].repeat(1, pad_length, 1)
            features_tensor = torch.cat([features_tensor, pad_features], dim=1)
            mask_tensor = torch.cat([mask_tensor, pad_mask], dim=1)

        batch_size, seq_len, num_features = features_tensor.shape

        member_scores: List[float] = []
        with torch.no_grad():
            for model in self.models:
                # Models only take (features, masks) - not delta_t
                logits = model(features_tensor, mask_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                member_scores.extend(probs.tolist())

        risk_score = float(np.mean(member_scores)) if member_scores else 0.5
        variance = float(np.var(member_scores)) if len(member_scores) > 1 else 0.0
        confidence = float(np.clip(1.0 - variance, 0.0, 1.0))

        # Risk level thresholds (configurable via env)
        try:
            high_t = float(os.getenv('DL_HIGH_THRESHOLD', '0.7'))
            med_t = float(os.getenv('DL_MEDIUM_THRESHOLD', '0.3'))
        except Exception:
            high_t, med_t = 0.7, 0.3

        # Use >= for consistency (threshold boundary should be included)
        if risk_score >= high_t:
            risk_level = 'High'
        elif risk_score >= med_t:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': confidence,
            'ensemble_members': len(self.models),
            'variance': variance
        }

class SepsisPredictionIntegration:
    """
    Comprehensive integration class that combines:
    - Person A's data preprocessing
    - Person B's baseline models (SIRS, qSOFA, NEWS2, SOFA, ML models)
    - Person C's deep learning models (GRU-D, LSTM, CNN-LSTM, Transformer)
    - Person D's explainability and dashboard
    """
    
    def __init__(self):
        """Initialize the integration system"""
        self.baseline_models = {}
        self.deep_learning_models = {}
        self.feature_names = []
        self.scaler = None              # Baseline scaler (for LR)
        self.dl_scaler = None           # Deep learning scaler (for LSTM/GRU/CNNLSTM/Transformer)
        self.baseline_imputer = None  # Imputer for baseline models
        self.feature_selector = None  # Feature selector for baseline models
        self.feature_defaults: Dict[str, float] = {}
        self.is_initialized = False
        # Allow temporarily disabling baseline models via env var (default: enabled now that models are trained)
        # Set DISABLE_BASELINES=1 to disable
        self.disable_baseline_models = str(os.getenv('DISABLE_BASELINES', '0')).lower() in ('1', 'true', 'yes')
        # Allow temporarily disabling deep learning models (default: enabled to use newly trained checkpoints)
        # Set DISABLE_DL=1 to disable
        self.disable_deep_learning_models = str(os.getenv('DISABLE_DL', '0')).lower() in ('1', 'true', 'yes')
        # Per-model thresholds loaded from training metrics (fallback to defaults if missing)
        self.baseline_thresholds: Dict[str, float] = {}

        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all models and preprocessing components"""
        try:
            print("🔄 Initializing Sepsis Prediction Integration System...")
            
            # Load feature names (from Person A/B data definitions)
            self._determine_feature_names()

            # Load default feature values from available datasets
            self._load_feature_defaults()

            # Load Person B's baseline models
            self._load_baseline_models()
            
            # Load Person C's deep learning models (if enabled)
            if not self.disable_deep_learning_models:
                self._load_deep_learning_models()
            else:
                print("⏸️ Deep learning models are DISABLED (set DISABLE_DL=0 to enable). Skipping load.")
            
            # Load preprocessing components
            self._load_preprocessing()
            
            # Load baseline imputer (for baseline models)
            self._load_baseline_imputer()
            
            # Load feature selector (if available)
            self._load_feature_selector()

            # Load per-model optimal thresholds for baseline models (if available)
            self._load_baseline_thresholds()
            
            self.is_initialized = True
            print("✅ Integration system initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.is_initialized = False

    def _determine_feature_names(self):
        """Determine feature names from available datasets."""
        # Prefer feature order from training config if available
        config_candidates = [
            'outputs/config.json',
            'Capstone/outputs/config.json'
        ]
        for cfg in config_candidates:
            if os.path.exists(cfg):
                try:
                    with open(cfg, 'r') as f:
                        cfg_json = json.load(f)
                        if 'feature_cols' in cfg_json and isinstance(cfg_json['feature_cols'], list):
                            self.feature_names = list(cfg_json['feature_cols'])
                            print(f"📋 Using feature order from {cfg}")
                            return
                except Exception as exc:
                    print(f"⚠️ Unable to load feature_cols from {cfg}: {exc}")
        dataset_candidates = [
            'Dataset.csv',
            'Capstone/Dataset.csv',
            'sample_patients_fixed.csv',
            'Capstone/sample_patients_fixed.csv',
            'sample_patients_correct.csv',
            'Capstone/sample_patients_correct.csv',
            'sample_patients.csv',
            'Capstone/sample_patients.csv'
        ]

        metadata_columns = {'Patient_ID', 'Time', 'Sepsis_Label', 'Unnamed: 0'}
        for path in dataset_candidates:
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, nrows=1)
                column_mapping = {}
                if 'Hour' in df.columns and 'Time' not in df.columns:
                    column_mapping['Hour'] = 'Time'
                if 'SepsisLabel' in df.columns and 'Sepsis_Label' not in df.columns:
                    column_mapping['SepsisLabel'] = 'Sepsis_Label'
                if column_mapping:
                    df = df.rename(columns=column_mapping)

                features = [col for col in df.columns if col not in metadata_columns]
                if features:
                    self.feature_names = features
                    return
            except Exception as exc:
                print(f"⚠️ Unable to infer feature names from {path}: {exc}")

        if not self.feature_names:
            self.feature_names = [
                'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
                'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
                'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
                'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
                'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb',
                'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender',
                'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
                'MAP_rolling_mean_6hr', 'MAP_delta_1hr', 'qSOFA_Resp', 'qSOFA_MAP',
                'qSOFA_Score_simple'
            ]

    def _load_feature_defaults(self):
        """Populate default values for features using available sample datasets."""
        candidate_paths = [
            'sample_patients_fixed.csv',
            'Capstone/sample_patients_fixed.csv',
            'sample_patients_correct.csv',
            'Capstone/sample_patients_correct.csv',
            'sample_patients.csv',
            'Capstone/sample_patients.csv',
            'Dataset.csv',
            'Capstone/Dataset.csv'
        ]

        defaults: Dict[str, float] = {}
        for path in candidate_paths:
            if not os.path.exists(path):
                continue
            try:
                # Use a subset of rows for very large datasets to reduce memory pressure
                read_kwargs = {'nrows': 5000} if 'Dataset.csv' in path else {}
                df = pd.read_csv(path, **read_kwargs)
                for feature in self.feature_names:
                    if feature in df.columns:
                        median_value = pd.to_numeric(df[feature], errors='coerce').median()
                        if not np.isnan(median_value):
                            defaults.setdefault(feature, float(median_value))
                if len(defaults) == len(self.feature_names):
                    break
            except Exception as exc:
                print(f"⚠️ Unable to derive defaults from {path}: {exc}")

        self.feature_defaults = defaults

    def _load_baseline_models(self):
        """Load Person B's baseline models"""
        try:
            if self.disable_baseline_models:
                print("⏸️ Baseline models are DISABLED (set DISABLE_BASELINES=0 to enable). Skipping load.")
                self.baseline_models = {}
                return
            # Possible filenames for baseline models saved by collaborators
            candidate_paths: Dict[str, List[str]] = {
                'logistic_regression': [
                    'outputs/models/logistic_regression.pkl',
                    'outputs/models/model_logistic.pkl',
                    'model_logistic.pkl',
                    'model_LR.pkl',  # Calibrated version from notebook
                    'outputs/models/model_LR.pkl'
                ],
                'random_forest': [
                    'outputs/models/random_forest.pkl',
                    'outputs/models/model_rf.pkl',
                    'model_rf.pkl'
                ],
                'xgboost': [
                    'outputs/models/xgboost.pkl',
                    'outputs/models/model_xgb.pkl',
                    'model_xgb_or_hgb.pkl'
                ]
            }

            for name, paths in candidate_paths.items():
                model = self._load_first_available_model(paths)
                if model is not None:
                    self.baseline_models[name] = model
                    print(f"✅ Loaded {name}")
                else:
                    print(f"⚠️ Trained artifact for {name} not found. Checked: {paths}")

            if not self.baseline_models:
                print("⚠️ No baseline model artifacts were loaded. Baseline predictions will be unavailable until artifacts are provided.")

        except Exception as e:
            print(f"⚠️ Error loading baseline models: {e}")

    def _load_first_available_model(self, paths: List[str]):
        """Helper to load the first existing model from a list of candidate paths."""
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                # Check if file is actually a pickle/joblib file by reading first few bytes
                try:
                    with open(path, 'rb') as f:
                        first_bytes = f.read(2)
                        # Valid pickle files start with specific bytes (protocol-dependent)
                        # Valid joblib files have specific headers
                        if len(first_bytes) < 2:
                            print(f"⚠️ File {path} appears to be empty or corrupted, skipping")
                            continue
                except Exception as check_exc:
                    print(f"⚠️ Cannot read file {path}: {check_exc}, skipping")
                    continue
                
                # Try joblib first (preferred for sklearn models)
                try:
                    model = joblib.load(path)
                    print(f"  ✅ Loaded with joblib from {path}")
                except Exception as joblib_exc:
                    # Fallback to pickle
                    try:
                        with open(path, 'rb') as f:
                            model = pickle.load(f)
                        print(f"  ✅ Loaded with pickle from {path}")
                    except Exception as pickle_exc:
                        # If both fail, skip this file and try next
                        print(f"⚠️ Failed loading model at {path} (joblib: {str(joblib_exc)[:50]}, pickle: {str(pickle_exc)[:50]}), trying next path")
                        continue
                
                # Log model type and calibration status
                model_type = type(model).__name__
                is_calibrated = hasattr(model, 'calibrated_classifiers_') or hasattr(model, 'calibrators_')
                print(f"  📦 Model type: {model_type}")
                if is_calibrated:
                    print(f"  ✅ Model is CALIBRATED")
                else:
                    print(f"  ⚠️ Model is NOT calibrated - may produce extreme probabilities")
                
                return model
            except Exception as exc:
                print(f"⚠️ Failed loading model at {path}: {exc}")
                continue
        return None
    
    def _infer_input_size_from_state_dict(self, state_dict: dict, model_type: str) -> Optional[int]:
        """Infer input size from state_dict weights"""
        try:
            if model_type == 'grud' or model_type == 'lstm':
                # For GRU/LSTM: weight_ih_l0 has shape [3*hidden_size, input_size]
                if 'gru.weight_ih_l0' in state_dict:
                    weight_shape = state_dict['gru.weight_ih_l0'].shape
                    if len(weight_shape) == 2:
                        return weight_shape[1]  # input_size is the second dimension
                elif 'lstm.weight_ih_l0' in state_dict:
                    weight_shape = state_dict['lstm.weight_ih_l0'].shape
                    if len(weight_shape) == 2:
                        return weight_shape[1]
                elif 'weight_ih_l0' in state_dict:
                    weight_shape = state_dict['weight_ih_l0'].shape
                    if len(weight_shape) == 2:
                        return weight_shape[1]
            elif model_type == 'transformer':
                # For Transformer: embedding or input_projection weight
                if 'input_projection.weight' in state_dict:
                    return state_dict['input_projection.weight'].shape[1]
                elif 'embedding.weight' in state_dict:
                    return state_dict['embedding.weight'].shape[1]
            elif model_type == 'cnn_lstm':
                # For CNN-LSTM: first conv layer
                for key in state_dict.keys():
                    if 'conv' in key and 'weight' in key:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) >= 2:
                            return weight_shape[1]  # input channels
        except Exception as e:
            print(f"  ⚠️ Could not infer input size: {e}")
        return None

    def _load_deep_learning_models(self):
        """Load Person C's deep learning models"""
        try:
            from models import GRUD, LSTM, CNNLSTM, Transformer

            # Try to load input_size from config file (models were trained with this size)
            input_size = None
            expected_seq_len = 24
            config_paths = [
                'outputs/config.json',
                'Capstone/outputs/config.json'
            ]
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            if 'n_features' in config:
                                input_size = config['n_features']
                                print(f"📋 Loaded input_size={input_size} from {config_path}")
                            if 'sequence_length' in config:
                                expected_seq_len = int(config['sequence_length'])
                                print(f"📋 Loaded sequence_length={expected_seq_len} from {config_path}")
                                break
                    except Exception as e:
                        print(f"⚠️ Could not read config from {config_path}: {e}")
            
            # Fallback to feature_names length if config not found
            if input_size is None:
                input_size = len(self.feature_names)
                print(f"⚠️ Using input_size={input_size} from feature_names (config not found)")

            model_configs = {
                'grud': {
                    'class': GRUD,
                    'init_args': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
                },
                'lstm': {
                    'class': LSTM,
                    'init_args': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3}
                },
                'cnn_lstm': {
                    'class': CNNLSTM,
                    'init_args': {'hidden_size': 128, 'num_layers': 1, 'dropout': 0.3,
                                  'cnn_filters': [64, 128], 'kernel_size': 3}
                },
                'transformer': {
                    'class': Transformer,
                    'init_args': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1}
                }
            }

            seeds = [42, 43, 44]

            for name, cfg in model_configs.items():
                models: List[torch.nn.Module] = []
                for seed in seeds:
                    model_path = os.path.join('outputs', 'models', f'{name}_seed{seed}.pt')
                    if not os.path.exists(model_path):
                        continue
                    try:
                        # Load the file
                        loaded_obj = torch.load(model_path, map_location='cpu', weights_only=False)
                        
                        # Check if it's a state_dict (OrderedDict/dict) or a full model
                        if isinstance(loaded_obj, torch.nn.Module):
                            # It's a full model, use it directly
                            model = loaded_obj
                            models.append(model)
                        elif isinstance(loaded_obj, dict):
                            # It's a state_dict, need to create model and load it
                            # First, try to infer the correct input_size from the state_dict
                            inferred_input_size = self._infer_input_size_from_state_dict(loaded_obj, name)
                            model_input_size = inferred_input_size if inferred_input_size else input_size
                            
                            if inferred_input_size and inferred_input_size != input_size:
                                print(f"  🔍 Inferred input_size={inferred_input_size} from checkpoint (config has {input_size})")
                            
                            try:
                                model = cfg['class'](input_size=model_input_size, **cfg['init_args'])
                                # Load only compatible tensor shapes to avoid size mismatch errors
                                try:
                                    model_state = model.state_dict()
                                    compatible = {k: v for k, v in loaded_obj.items()
                                                  if k in model_state and hasattr(model_state[k], 'shape') and hasattr(v, 'shape')
                                                  and model_state[k].shape == v.shape}
                                    model.load_state_dict(compatible, strict=False)
                                    _warn_if_partial_load(f'{name}', model_state, compatible)
                                except Exception as load_exc:
                                    print(f"  ⚠️ Safe state_dict load failed for {name}_seed{seed}: {load_exc}")
                                    continue
                                models.append(model)
                            except (RuntimeError, KeyError, TypeError) as e:
                                print(f"  ⚠️ Could not load state_dict for {name}_seed{seed}: {e}")
                                continue
                        else:
                            print(f"  ⚠️ Unexpected object type for {name}_seed{seed}: {type(loaded_obj)}")
                            continue
                    except Exception as e:
                        print(f"⚠️ Error loading {name}_seed{seed}: {e}")

                if not models:
                    # Try real_data model first (preferred)
                    fallback_paths = [
                        os.path.join('outputs', 'models', f'{name}_real_data.pt'),
                        os.path.join('outputs', 'models', f'{name}_demo_model.pt')
                    ]
                    for fallback_path in fallback_paths:
                        if os.path.exists(fallback_path):
                            try:
                                # Load the file
                                loaded_obj = torch.load(fallback_path, map_location='cpu', weights_only=False)
                                
                                # Check if it's a state_dict (OrderedDict/dict) or a full model
                                if isinstance(loaded_obj, torch.nn.Module):
                                    # It's a full model, use it directly
                                    model = loaded_obj
                                    models.append(model)
                                elif isinstance(loaded_obj, dict):
                                    # It's a state_dict, need to create model and load it
                                    # First, try to infer the correct input_size from the state_dict
                                    inferred_input_size = self._infer_input_size_from_state_dict(loaded_obj, name)
                                    model_input_size = inferred_input_size if inferred_input_size else input_size
                                    
                                    if inferred_input_size and inferred_input_size != input_size:
                                        print(f"  🔍 Inferred input_size={inferred_input_size} from checkpoint (config has {input_size})")
                                    
                                    try:
                                        model = cfg['class'](input_size=model_input_size, **cfg['init_args'])
                                        # Load only compatible tensor shapes to avoid size mismatch errors
                                        try:
                                            model_state = model.state_dict()
                                            compatible = {k: v for k, v in loaded_obj.items()
                                                          if k in model_state and hasattr(model_state[k], 'shape') and hasattr(v, 'shape')
                                                          and model_state[k].shape == v.shape}
                                            model.load_state_dict(compatible, strict=False)
                                            _warn_if_partial_load(f'{name}', model_state, compatible)
                                        except Exception as load_exc:
                                            print(f"  ⚠️ Safe state_dict load failed from {os.path.basename(fallback_path)}: {load_exc}")
                                            continue
                                        models.append(model)
                                    except (RuntimeError, KeyError, TypeError) as e:
                                        print(f"  ⚠️ Could not load state_dict from {os.path.basename(fallback_path)}: {e}")
                                        continue
                                else:
                                    print(f"  ⚠️ Unexpected object type from {os.path.basename(fallback_path)}: {type(loaded_obj)}")
                                    continue
                                
                                model_type = "real_data" if "real_data" in fallback_path else "demo"
                                print(f"  📦 Loaded {model_type} model from {os.path.basename(fallback_path)}")
                                break
                            except Exception as e:
                                print(f"⚠️ Error loading {fallback_path}: {e}")
                                import traceback
                                traceback.print_exc()

                if models:
                    # Determine the actual input size used by the loaded models
                    # Check the first model to see what input size it actually has
                    ensemble_input_size = input_size
                    if models and name != 'cnn_lstm':
                        first_model = models[0]
                        try:
                            # Try to infer from model parameters
                            for param_name, param in first_model.named_parameters():
                                if 'weight_ih_l0' in param_name or 'input_projection.weight' in param_name:
                                    if len(param.shape) == 2:
                                        inferred = param.shape[1]
                                        if inferred != input_size:
                                            ensemble_input_size = inferred
                                            print(f"  🔍 Using input_size={ensemble_input_size} from loaded model (config had {input_size})")
                                        break
                        except Exception:
                            pass  # Use default input_size
                    # For CNN-LSTM, keep feature dimension from config/state_dict (do not overwrite with LSTM hidden dims)
                    if name == 'cnn_lstm':
                        ensemble_input_size = input_size
                    
                    self.deep_learning_models[name] = TorchModelEnsemble(name, models, ensemble_input_size, expected_seq_len)
                    print(f"✅ Loaded {name} ({len(models)} checkpoint(s))")
                else:
                    print(f"⚠️ No checkpoints found for {name}")

        except Exception as e:
            print(f"⚠️ Error loading deep learning models: {e}")
            import traceback
            traceback.print_exc()

    def _load_dl_weights_from_results(self) -> Dict[str, float]:
        """Load per-model weights from results CSVs using AUPRC; normalize to sum=1.
        Prefers validation (model_comparison.csv), falls back to test (test_set_model_comparison.csv).
        """
        candidates = [
            os.path.join('outputs', 'results', 'model_comparison.csv'),
            os.path.join('outputs', 'results', 'test_set_model_comparison.csv')
        ]
        weights: Dict[str, float] = {}
        # Aggregate from all available files to maximize coverage
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                import csv
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = row.get('model', '').strip().lower()
                        if not name:
                            continue
                        try:
                            auprc = float(row.get('auprc', '0') or 0)
                        except Exception:
                            auprc = 0.0
                        if auprc > 0:
                            # Keep the max AUPRC seen across files for that model
                            prev = weights.get(name, 0.0)
                            if auprc > prev:
                                weights[name] = auprc
            except Exception as exc:
                print(f"⚠️ Failed reading results from {path}: {exc}")
        if not weights:
            return {}
        # Map to integration model keys if needed
        key_map = {
            'grud': 'grud',
            'lstm': 'lstm',
            'cnn_lstm': 'cnn_lstm',
            'cnn-lstm': 'cnn_lstm',
            'transformer': 'transformer'
        }
        mapped: Dict[str, float] = {}
        for k, v in weights.items():
            mk = key_map.get(k, k)
            mapped[mk] = v
        # Filter to only models that are actually loaded
        if hasattr(self, 'deep_learning_models') and isinstance(self.deep_learning_models, dict):
            loaded = set(self.deep_learning_models.keys())
            mapped = {k: v for k, v in mapped.items() if k in loaded}
        # If we still have < 2 models, better to fallback in the caller to equal weights
        total = sum(mapped.values()) or 0.0
        if total <= 0.0:
            return {}
        return {k: float(v / total) for k, v in mapped.items()}

    def _load_preprocessing(self):
        """Load preprocessing components"""
        try:
            # Combined logic: Try several candidate paths, support both .pkl and .joblib
            baseline_scaler_candidates = [
                'outputs/models/scaler.pkl',            # Baseline (LR) scaler
                'outputs/models/scaler.joblib',
                'outputs/cache/scaler.joblib',
                'scaler_lr.pkl',
                'Capstone/outputs/models/scaler.pkl'
            ]
            dl_scaler_candidates = [
                'outputs/scaler_final.pkl',            # Prefer DL scaler if available
                'Capstone/outputs/scaler_final.pkl'
            ]

            # Load baseline scaler
            self.scaler = None
            for path in baseline_scaler_candidates:
                if not os.path.exists(path):
                    continue
                try:
                    # Check if file is readable and not empty
                    try:
                        with open(path, 'rb') as f:
                            first_bytes = f.read(2)
                            if len(first_bytes) < 2:
                                print(f"⚠️ File {path} appears to be empty or corrupted, skipping")
                                continue
                    except Exception as check_exc:
                        print(f"⚠️ Cannot read file {path}: {check_exc}, skipping")
                        continue
                    
                    # Try joblib first (preferred for sklearn objects)
                    try:
                        self.scaler = joblib.load(path)
                        print(f"✅ Loaded scaler with joblib from {path}")
                    except Exception as joblib_exc:
                        # Fallback to pickle
                        try:
                            with open(path, 'rb') as f:
                                self.scaler = pickle.load(f)
                            print(f"✅ Loaded scaler with pickle from {path}")
                        except Exception as pickle_exc:
                            # If both fail, skip this file and try next
                            print(f"⚠️ Failed loading scaler at {path} (joblib: {str(joblib_exc)[:50]}, pickle: {str(pickle_exc)[:50]}), trying next path")
                            continue
                    
                    # Check if scaler is fitted (has mean_)
                    if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                        print(f"✅ Scaler is fitted (mean shape: {self.scaler.mean_.shape})")
                    else:
                        print(f"⚠️ Loaded scaler from {path} but it's not fitted")
                    break
                except Exception as exc:
                    print(f"⚠️ Failed loading scaler at {path}: {exc}")
                    continue

            if self.scaler is None:
                self.scaler = StandardScaler()
                print("⚠️ No pretrained baseline scaler found. A new StandardScaler will be fitted on incoming data.")

            # Load deep learning scaler (prefer DL-specific scaler)
            self.dl_scaler = None
            for path in dl_scaler_candidates:
                if not os.path.exists(path):
                    continue
                try:
                    # Check if file is readable and not empty
                    try:
                        with open(path, 'rb') as f:
                            first_bytes = f.read(2)
                            if len(first_bytes) < 2:
                                print(f"⚠️ File {path} appears to be empty or corrupted, skipping")
                                continue
                    except Exception as check_exc:
                        print(f"⚠️ Cannot read file {path}: {check_exc}, skipping")
                        continue
                    
                    try:
                        self.dl_scaler = joblib.load(path)
                        print(f"✅ Loaded DL scaler from {path}")
                        break
                    except Exception as joblib_exc:
                        # Try pickle as fallback
                        try:
                            with open(path, 'rb') as f:
                                self.dl_scaler = pickle.load(f)
                            print(f"✅ Loaded DL scaler with pickle from {path}")
                            break
                        except Exception as pickle_exc:
                            print(f"⚠️ Failed loading DL scaler at {path} (joblib: {str(joblib_exc)[:50]}, pickle: {str(pickle_exc)[:50]}), trying next path")
                            continue
                except Exception as exc:
                    print(f"⚠️ Failed loading DL scaler at {path}: {exc}")
                    continue

            if self.dl_scaler is None:
                # Fallback to baseline scaler for DL if no DL scaler available
                self.dl_scaler = self.scaler
                print("⚠️ No DL-specific scaler found. Using baseline scaler for deep learning models.")
        except Exception as e:
            print(f"⚠️ Error loading preprocessing: {e}")
            self.scaler = StandardScaler()
            self.dl_scaler = self.scaler
    
    def _load_baseline_imputer(self):
        """Load imputer for baseline models"""
        try:
            imputer_candidates = [
                'outputs/models/baseline_imputer.pkl',
                'outputs/cache/imputer.joblib',
                'Capstone/outputs/models/baseline_imputer.pkl'
            ]
            
            self.baseline_imputer = None
            for path in imputer_candidates:
                if not os.path.exists(path):
                    continue
                try:
                    # Check if file is readable and not empty
                    try:
                        with open(path, 'rb') as f:
                            first_bytes = f.read(2)
                            if len(first_bytes) < 2:
                                print(f"⚠️ File {path} appears to be empty or corrupted, skipping")
                                continue
                    except Exception as check_exc:
                        print(f"⚠️ Cannot read file {path}: {check_exc}, skipping")
                        continue
                    
                    try:
                        self.baseline_imputer = joblib.load(path)
                        print(f"✅ Loaded baseline imputer from {path}")
                        break
                    except Exception as joblib_exc:
                        # Try pickle as fallback
                        try:
                            with open(path, 'rb') as f:
                                self.baseline_imputer = pickle.load(f)
                            print(f"✅ Loaded baseline imputer with pickle from {path}")
                            break
                        except Exception as pickle_exc:
                            print(f"⚠️ Failed loading baseline imputer at {path} (joblib: {str(joblib_exc)[:50]}, pickle: {str(pickle_exc)[:50]}), trying next path")
                            continue
                except Exception as exc:
                    print(f"⚠️ Failed loading baseline imputer at {path}: {exc}")
                    continue
            
            if self.baseline_imputer is None:
                print("⚠️ No baseline imputer found. Will use default value filling.")
        except Exception as e:
            print(f"⚠️ Error loading baseline imputer: {e}")

    def _load_feature_selector(self):
        """Load feature selector for baseline models (if trained)."""
        try:
            selector_candidates = [
                'outputs/models/feature_selector.pkl',
                'outputs/cache/feature_selector.pkl',
                'Capstone/outputs/models/feature_selector.pkl'
            ]
            self.feature_selector = None
            for path in selector_candidates:
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'rb') as f:
                        first_bytes = f.read(2)
                        if len(first_bytes) < 2:
                            print(f"⚠️ File {path} appears to be empty or corrupted, skipping")
                            continue
                except Exception as check_exc:
                    print(f"⚠️ Cannot read file {path}: {check_exc}, skipping")
                    continue

                try:
                    self.feature_selector = joblib.load(path)
                    print(f"✅ Loaded feature selector from {path}")
                    break
                except Exception as joblib_exc:
                    try:
                        with open(path, 'rb') as f:
                            self.feature_selector = pickle.load(f)
                        print(f"✅ Loaded feature selector with pickle from {path}")
                        break
                    except Exception as pickle_exc:
                        print(f"⚠️ Failed loading feature selector at {path} (joblib: {str(joblib_exc)[:50]}, pickle: {str(pickle_exc)[:50]}), trying next path")
                        continue

            if self.feature_selector is None:
                print("ℹ️ No baseline feature selector found.")
        except Exception as e:
            print(f"⚠️ Error loading feature selector: {e}")
    
    def preprocess_patient_data(self, patient_data):
        """Preprocess patient data using Person A's methods"""
        try:
            # Convert to DataFrame if not already
            if isinstance(patient_data, dict):
                df = pd.DataFrame([patient_data])
            else:
                df = patient_data.copy()

            # If Series, convert to DataFrame
            if isinstance(df, pd.Series):
                df = df.to_frame().T

            df = df.reset_index(drop=True)

            # Normalize common input keys to training feature names (case-insensitive for sources)
            key_mapping = {
                'heart_rate': 'HR', 'hr': 'HR',
                'respiratory_rate': 'Resp', 'rr': 'Resp',
                'temperature': 'Temp', 'temp': 'Temp',
                'oxygen_saturation': 'O2Sat', 'spo2': 'O2Sat',
                'map': 'MAP', 'sbp': 'SBP', 'dbp': 'DBP',
                'lactate': 'Lactate', 'wbc': 'WBC', 'creatinine': 'Creatinine',
                'bilirubin_total': 'Bilirubin_total', 'bilirubin': 'Bilirubin_total',
                'age': 'Age', 'gender': 'Gender', 'platelets': 'Platelets',
                'iculos': 'ICULOS'
            }
            # Build lowercase column lookup
            col_lower_to_orig = {c.lower(): c for c in df.columns}
            for src_lower, target in key_mapping.items():
                if src_lower in col_lower_to_orig:
                    src_col = col_lower_to_orig[src_lower]
                    if target not in df.columns:
                        df[target] = pd.to_numeric(df[src_col], errors='coerce') if target not in ['Gender'] else df[src_col]

            # Handle categorical columns (Gender, Unit1, Unit2)
            if 'Gender' in df.columns:
                # Convert Gender to numeric: Male=1, Female=0
                df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0)

            if 'Unit1' in df.columns:
                df['Unit1'] = pd.to_numeric(df['Unit1'], errors='coerce').fillna(0)

            if 'Unit2' in df.columns:
                df['Unit2'] = pd.to_numeric(df['Unit2'], errors='coerce').fillna(0)

            # Default values for common features (used when feature is missing)
            # If class/instance has self.feature_defaults, prefer it, else fallback defaults
            default_values = getattr(self, 'feature_defaults', None)
            if default_values is None:
                default_values = {
                    'HR': 80, 'O2Sat': 95, 'Temp': 37, 'SBP': 120, 'MAP': 75, 'DBP': 80,
                    'Resp': 16, 'EtCO2': 40, 'BaseExcess': 0, 'HCO3': 24, 'FiO2': 21,
                    'pH': 7.4, 'PaCO2': 40, 'SaO2': 95, 'AST': 30, 'BUN': 15,
                    'Alkalinephos': 100, 'Calcium': 9, 'Chloride': 100, 'Creatinine': 1.0,
                    'Bilirubin_direct': 0.2, 'Lactate': 1.0, 'Magnesium': 2.0, 'Phosphate': 3.5,
                    'Potassium': 4.0, 'Bilirubin_total': 1.0, 'TroponinI': 0.01, 'Hct': 40,
                    'Hgb': 12, 'PTT': 30, 'WBC': 8, 'Fibrinogen': 300, 'Platelets': 250,
                    'Age': 65, 'Gender': 0, 'Unit1': 0, 'Unit2': 0, 'HospAdmTime': 0, 'ICULOS': 0
                }

            # Build features and observation mask in the same way as the HEAD version
            feature_columns: Dict[str, pd.Series] = {}
            for feature in self.feature_names:
                if feature in df.columns:
                    series = pd.to_numeric(df[feature], errors='coerce')
                else:
                    series = pd.Series(np.nan, index=df.index, dtype=float)
                feature_columns[feature] = series

            df_features = pd.DataFrame(feature_columns)
            observation_mask = (~df_features.isna()).astype(np.float32).values
            provided_count = int(np.sum(observation_mask)) if observation_mask.size else 0

            for feature in self.feature_names:
                default_value = default_values.get(feature, 0.0)
                df_features[feature].fillna(default_value, inplace=True)

            feature_matrix = df_features.astype(np.float32).values

            # Optional debug to detect mostly-default inputs leading to similar predictions
            try:
                debug_flag = str(os.getenv('DEBUG_INFERENCE', '0')).lower() in ('1', 'true', 'yes')
                if debug_flag:
                    # Count provided (non-NaN before filling) per timestep if sequence
                    provided_per_row = observation_mask.sum(axis=1).tolist() if observation_mask.ndim == 2 else [provided_count]
                    print(f"🔎 DEBUG INFERENCE: provided_features_per_row={provided_per_row} of {len(self.feature_names)}")
            except Exception:
                pass

            # Scale features for DL using DL-specific scaler if available and fitted
            if self.dl_scaler is not None:
                try:
                    # Check if scaler is fitted
                    is_fitted = hasattr(self.dl_scaler, 'mean_') and self.dl_scaler.mean_ is not None

                    if is_fitted:
                        # Check if scaler expects the same number of features
                        if hasattr(self.dl_scaler, 'n_features_in_'):
                            expected_features = self.dl_scaler.n_features_in_
                            if feature_matrix.shape[1] != expected_features:
                                # Try to use only the first N features if we have more
                                if feature_matrix.shape[1] > expected_features:
                                    feature_matrix = feature_matrix[:, :expected_features]
                                else:
                                    # Pad with zeros if we have fewer features
                                    padding = np.zeros((feature_matrix.shape[0], expected_features - feature_matrix.shape[1]))
                                    feature_matrix = np.hstack([feature_matrix, padding])
                        scaled_features = self.dl_scaler.transform(feature_matrix)
                        try:
                            if debug_flag:
                                # Report simple statistics of scaled features for first row
                                row0 = scaled_features[0] if scaled_features.ndim == 2 else scaled_features
                                print(f"🔎 DEBUG INFERENCE: scaled_row0_mean={float(np.mean(row0)):.3f}, std={float(np.std(row0)):.3f}")
                        except Exception:
                            pass
                    else:
                        # Scaler not fitted - avoid fitting on single sample
                        scaled_features = feature_matrix
                except Exception as exc:
                    print(f"⚠️ Falling back to unscaled features: {exc}")
                    scaled_features = feature_matrix
            else:
                scaled_features = feature_matrix

            # For compatibility, return feature names as the third return value
            return scaled_features.astype(np.float32), observation_mask, self.feature_names

        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []

    
    def predict_baseline_models(self, patient_data):
        """Get predictions from Person B's baseline models"""
        predictions = {}
        if getattr(self, 'disable_baseline_models', False):
            print("BASELINE DEBUG: Baselines disabled, skipping predictions.")
            # Baseline models are temporarily disabled
            return predictions
        
        try:
            # Preprocess data for baseline models
            # Note: Different models need different preprocessing:
            # - Logistic Regression: imputation -> scaling
            # - Random Forest: imputation only (no scaling)
            # - XGBoost: imputation only (no scaling)
            
            # Prepare scaled + feature-selected data for all baseline models (matches training)
            X_scaled = self._preprocess_for_baseline_models(patient_data, apply_scaling=True)
            if X_scaled is None:
                print("BASELINE DEBUG: Preprocessing returned None.")
                return predictions
            else:
                try:
                    print(f"BASELINE DEBUG: X_scaled shape={getattr(X_scaled, 'shape', None)}")
                except Exception:
                    pass

            # Get predictions from each baseline model
            for name, model in self.baseline_models.items():
                try:
                    # All baseline models use scaled + feature-selected data
                    X_for_model = X_scaled.copy()
                    
                    # Check what features the model expects
                    if hasattr(model, 'n_features_in_'):
                        expected_features = model.n_features_in_
                        current_features = X_for_model.shape[1]
                        
                        if current_features != expected_features:
                            print(f"BASELINE DEBUG: {name} expected {expected_features} features, got {current_features}. Adjusting.")
                            if current_features > expected_features:
                                # Use only the first N features
                                X_for_model = X_for_model[:, :expected_features]
                            else:
                                # Pad with zeros
                                padding = np.zeros((X_for_model.shape[0], expected_features - current_features))
                                X_for_model = np.hstack([X_for_model, padding])
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_for_model)[0]
                        prob = proba[1] if len(proba) > 1 else proba[0]
                        try:
                            print(f"BASELINE DEBUG: {name} prob={float(prob):.4f}")
                        except Exception:
                            pass
                        # Calculate confidence based on probability distribution
                        # Higher confidence when probability is more extreme (closer to 0 or 1)
                        # Also consider the difference between class probabilities
                        if len(proba) > 1:
                            max_prob = max(proba)
                            # Confidence increases with distance from 0.5 and with class separation
                            confidence = float(np.clip(
                                0.5 + abs(prob - 0.5) * 0.8 + (max_prob - 0.5) * 0.3,
                                0.5, 0.95
                            ))
                        else:
                            # Fallback if only one probability
                            confidence = float(np.clip(0.5 + abs(prob - 0.5) * 0.8, 0.5, 0.95))
                    else:
                        prob = model.predict(X_for_model)[0]
                        # If predict returns binary, convert to probability-like score
                        if isinstance(prob, (int, np.integer)):
                            prob = float(prob)
                            # For binary predictions, lower confidence since we don't have probability distribution
                            confidence = 0.65
                        elif prob > 1.0:
                            prob = prob / 100.0
                            confidence = float(np.clip(0.5 + abs(prob - 0.5) * 0.6, 0.5, 0.9))
                        else:
                            # Assume it's already a probability
                            confidence = float(np.clip(0.5 + abs(prob - 0.5) * 0.8, 0.5, 0.95))
                    
                    # Cap extreme probabilities to prevent perfect scores
                    # This helps identify potential calibration issues
                    if prob >= 0.999:
                        import warnings
                        warnings.warn(f"⚠️ {name} model output extreme probability: {prob:.6f}. "
                                    f"This may indicate calibration issues or data problems.")
                        # Cap at 0.99 for display, but keep original for logging
                        prob_display = 0.99
                    else:
                        prob_display = prob
                    
                    predictions[name] = {
                        'risk_score': prob_display,  # Use capped value for display
                        'risk_score_raw': prob,  # Keep original for debugging
                        'risk_level': (
                            'High'
                            if prob_display >= self.baseline_thresholds.get(name, 0.7)
                            else 'Medium'
                            if prob_display >= float(os.getenv('BASELINE_MEDIUM_THRESHOLD', '0.3'))
                            else 'Low'
                        ),
                        'confidence': confidence,
                        'extreme_probability_warning': prob >= 0.999,  # Flag for dashboard
                        'threshold_used': self.baseline_thresholds.get(name, 0.7)
                    }
                except Exception as e:
                    print(f"⚠️ Error with {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    predictions[name] = {
                        'risk_score': 0.5,
                        'risk_level': 'Medium',
                        'confidence': 0.5
                    }
            
            # Calculate clinical scores
            predictions['clinical_scores'] = self._calculate_clinical_scores(patient_data)
            # Debug summary
            keys_now = [k for k in predictions.keys() if k != 'clinical_scores']
            print(f"BASELINE DEBUG: predictions keys={keys_now}")
            
        except Exception as e:
            print(f"❌ Error in baseline predictions: {e}")
        
        return predictions
    
    def _preprocess_for_baseline_models(self, patient_data, apply_scaling: bool = True):
        """
        Preprocess patient data specifically for baseline models.
        
        Args:
            patient_data: Patient data (dict, Series, or DataFrame)
            apply_scaling: If True, apply scaling after imputation. 
                          Logistic Regression needs scaling, RF/XGBoost don't.
        
        Returns:
            Preprocessed feature matrix (imputed, and optionally scaled)
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(patient_data, dict):
                df = pd.DataFrame([patient_data])
            elif isinstance(patient_data, pd.Series):
                df = patient_data.to_frame().T
            else:
                df = patient_data.copy()
            
            df = df.reset_index(drop=True)
            
            # Handle categorical columns
            if 'Gender' in df.columns:
                df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0)
            if 'Unit1' in df.columns:
                df['Unit1'] = pd.to_numeric(df['Unit1'], errors='coerce').fillna(0)
            if 'Unit2' in df.columns:
                df['Unit2'] = pd.to_numeric(df['Unit2'], errors='coerce').fillna(0)
            
            # Compute clinical score features exactly as used during baseline training
            # (SIRS_count, SIRS_flag, qSOFA, NEWS2, SOFA_partial)
            try:
                def _compute_sirs(row):
                    cnt = 0
                    t = row.get('Temp', np.nan)
                    if pd.notna(t) and (t > 38.0 or t < 36.0):
                        cnt += 1
                    hr = row.get('HR', np.nan)
                    if pd.notna(hr) and hr > 90:
                        cnt += 1
                    rr = row.get('Resp', np.nan)
                    if pd.notna(rr) and rr > 20:
                        cnt += 1
                    wbc = row.get('WBC', np.nan)
                    if pd.notna(wbc) and (wbc > 12 or wbc < 4):
                        cnt += 1
                    return cnt

                def _compute_qsofa(row):
                    if 'qSOFA_Score_simple' in row.index:
                        v = row.get('qSOFA_Score_simple', np.nan)
                        if pd.notna(v):
                            try:
                                return int(v)
                            except Exception:
                                pass
                    rr = row.get('Resp', np.nan)
                    sbp = row.get('SBP', np.nan)
                    score = 0
                    if pd.notna(rr) and rr >= 22:
                        score += 1
                    if pd.notna(sbp) and sbp <= 100:
                        score += 1
                    return score

                def _compute_news2(row):
                    score = 0
                    rr = row.get('Resp', np.nan)
                    if pd.notna(rr):
                        if rr <= 8:
                            score += 3
                        elif 9 <= rr <= 11:
                            score += 1
                        elif 21 <= rr <= 24:
                            score += 2
                        elif rr >= 25:
                            score += 3
                    spo2 = row.get('O2Sat', row.get('SaO2', np.nan))
                    if pd.notna(spo2):
                        if spo2 <= 91:
                            score += 3
                        elif 92 <= spo2 <= 93:
                            score += 2
                        elif 94 <= spo2 <= 95:
                            score += 1
                    sbp = row.get('SBP', np.nan)
                    if pd.notna(sbp):
                        if sbp <= 90:
                            score += 3
                        elif 91 <= sbp <= 100:
                            score += 2
                        elif 101 <= sbp <= 110:
                            score += 1
                    hr = row.get('HR', np.nan)
                    if pd.notna(hr):
                        if hr <= 40:
                            score += 3
                        elif 41 <= hr <= 50:
                            score += 1
                        elif 91 <= hr <= 110:
                            score += 1
                        elif 111 <= hr <= 130:
                            score += 2
                        elif hr >= 131:
                            score += 3
                    temp = row.get('Temp', np.nan)
                    if pd.notna(temp):
                        if temp <= 35.0:
                            score += 3
                        elif 35.1 <= temp <= 36.0:
                            score += 1
                        elif 38.1 <= temp <= 39.0:
                            score += 1
                        elif temp >= 39.1:
                            score += 2
                    return score

                def _compute_sofa_partial(row):
                    score = 0
                    p = row.get('Platelets', np.nan)
                    if pd.notna(p):
                        if p >= 150:
                            score += 0
                        elif 100 <= p < 150:
                            score += 1
                        elif 50 <= p < 100:
                            score += 2
                        elif 20 <= p < 50:
                            score += 3
                        else:
                            score += 4
                    b = row.get('Bilirubin_total', np.nan)
                    if pd.notna(b):
                        if b < 1.2:
                            score += 0
                        elif b < 2.0:
                            score += 1
                        elif b < 6.0:
                            score += 2
                        elif b < 12.0:
                            score += 3
                        else:
                            score += 4
                    cr = row.get('Creatinine', np.nan)
                    if pd.notna(cr):
                        if cr < 1.2:
                            score += 0
                        elif cr < 2.0:
                            score += 1
                        elif cr < 3.5:
                            score += 2
                        elif cr < 5.0:
                            score += 3
                        else:
                            score += 4
                    mapv = row.get('MAP', np.nan)
                    if pd.notna(mapv):
                        if mapv >= 70:
                            score += 0
                        elif 50 <= mapv < 70:
                            score += 1
                        else:
                            score += 2
                    return score

                # Compute into the working dataframe
                df['SIRS_count'] = df.apply(_compute_sirs, axis=1)
                df['SIRS_flag'] = (df['SIRS_count'] >= 2).astype(int)
                df['qSOFA'] = df.apply(_compute_qsofa, axis=1)
                df['NEWS2'] = df.apply(_compute_news2, axis=1)
                df['SOFA_partial'] = df.apply(_compute_sofa_partial, axis=1)
            except Exception as e:
                print(f"⚠️ Failed computing clinical score features for baselines: {e}")
                # Proceed; imputer will fill missing
            
            # Build feature matrix using baseline training feature order
            def _baseline_feature_order() -> List[str]:
                # Matches actual features used in training (Engineered: 0)
                base_features = [
                    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'FiO2', 'pH', 'PaCO2',
                    'SaO2', 'BaseExcess', 'HCO3', 'WBC', 'Platelets', 'Creatinine',
                    'Bilirubin_total', 'Age'
                ]
                clinical_scores = ['SIRS_count', 'SIRS_flag', 'qSOFA', 'NEWS2', 'SOFA_partial']
                # Return exactly 23 features used by scaler/imputer
                return base_features + clinical_scores

            baseline_features = _baseline_feature_order()
            feature_columns = {}
            for feature in baseline_features:
                if feature in df.columns:
                    series = pd.to_numeric(df[feature], errors='coerce')
                else:
                    series = pd.Series(np.nan, index=df.index, dtype=float)
                feature_columns[feature] = series
            
            df_features = pd.DataFrame(feature_columns)
            feature_matrix = df_features.values
            
            # Apply imputation (matching training: median strategy)
            if self.baseline_imputer is not None:
                try:
                    feature_matrix = self.baseline_imputer.transform(feature_matrix)
                except Exception as e:
                    print(f"⚠️ Imputation failed, using defaults: {e}")
                    # Fallback to default values aligned to baseline_features
                    defaults = getattr(self, 'feature_defaults', {}) or {}
                    for i, feature in enumerate(baseline_features):
                        default_val = defaults.get(feature, 0.0)
                        feature_matrix[:, i] = np.nan_to_num(feature_matrix[:, i], nan=default_val)
            else:
                # No imputer available, use default values
                defaults = getattr(self, 'feature_defaults', {}) or {}
                for i, feature in enumerate(baseline_features):
                    default_val = defaults.get(feature, 0.0)
                    feature_matrix[:, i] = np.nan_to_num(feature_matrix[:, i], nan=default_val)
            
            # Apply scaling if requested
            if apply_scaling and self.scaler is not None:
                try:
                    is_fitted = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
                    if is_fitted:
                        # Handle feature dimension mismatch
                        if hasattr(self.scaler, 'n_features_in_'):
                            expected_features = self.scaler.n_features_in_
                            if feature_matrix.shape[1] != expected_features:
                                if feature_matrix.shape[1] > expected_features:
                                    feature_matrix = feature_matrix[:, :expected_features]
                                else:
                                    padding = np.zeros((feature_matrix.shape[0], expected_features - feature_matrix.shape[1]))
                                    feature_matrix = np.hstack([feature_matrix, padding])
                        feature_matrix = self.scaler.transform(feature_matrix)
                except Exception as e:
                    print(f"⚠️ Scaling failed: {e}")
            
            # Apply feature selection if available (after scaling)
            try:
                if apply_scaling and self.feature_selector is not None and hasattr(self.feature_selector, 'transform'):
                    feature_matrix = self.feature_selector.transform(feature_matrix)
            except Exception as e:
                print(f"⚠️ Feature selection transform failed: {e}")
            
            return feature_matrix.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error preprocessing for baseline models: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_baseline_thresholds(self):
        """Load per-model optimal thresholds learned during training."""
        try:
            candidates = [
                os.path.join('outputs', 'models', 'baseline_models_metrics.json'),
                'baseline_models_metrics.json',
                os.path.join('Capstone', 'outputs', 'models', 'baseline_models_metrics.json')
            ]
            metrics = None
            for path in candidates:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            metrics = json.load(f)
                        print(f"✅ Loaded baseline thresholds from {path}")
                        break
                    except Exception as e:
                        print(f"⚠️ Could not read thresholds from {path}: {e}")
                        continue
            if isinstance(metrics, dict):
                for key in ('logistic_regression', 'random_forest', 'xgboost'):
                    try:
                        th = metrics.get(key, {}).get('optimal_threshold', None)
                        if th is not None:
                            self.baseline_thresholds[key] = float(th)
                    except Exception:
                        continue
            # Fallback defaults if none loaded
            if not self.baseline_thresholds:
                self.baseline_thresholds = {}
                print("ℹ️ No per-model thresholds found; using defaults (High=0.7, Medium=0.3).")
        except Exception as e:
            print(f"⚠️ Error loading baseline thresholds: {e}")
    
    def predict_deep_learning_models(self, patient_data):
        """Get predictions from Person C's deep learning models"""
        predictions = {}
        if getattr(self, 'disable_deep_learning_models', False):
            print("DL DEBUG: Deep learning models disabled, skipping predictions.")
            return predictions
        
        try:
            # Preprocess data
            X_processed, observation_mask, _ = self.preprocess_patient_data(patient_data)

            if X_processed is None or observation_mask is None:
                return predictions

            # Get predictions from each deep learning model
            for name, model in self.deep_learning_models.items():
                try:
                    if hasattr(model, 'predict'):
                        seq_features = X_processed[np.newaxis, ...]
                        seq_mask = observation_mask[np.newaxis, ...]
                        predictions[name] = model.predict(seq_features, seq_mask)
                    else:
                        print(f"⚠️ Loaded deep learning component for {name} lacks predict method")
                except Exception as e:
                    print(f"⚠️ Error with {name}: {e}")
                    predictions[name] = {
                        'risk_score': 0.5,
                        'risk_level': 'Medium',
                        'confidence': 0.5
                    }
        
        except Exception as e:
            print(f"❌ Error in deep learning predictions: {e}")
        
        return predictions
    
    def _calculate_clinical_scores(self, patient_data):
        """Calculate clinical scores (Person B's implementation)"""
        scores = {}

        try:
            if isinstance(patient_data, pd.DataFrame):
                record = patient_data.iloc[-1].to_dict()
            elif isinstance(patient_data, dict):
                record = patient_data
            else:
                try:
                    record = dict(patient_data)
                except Exception:
                    record = {}

            # SIRS Score
            sirs_score = 0
            temp = record.get('Temp', 37)
            if temp > 38 or temp < 36:
                sirs_score += 1
            if record.get('HR', 80) > 90:
                sirs_score += 1
            if record.get('Resp', 16) > 20:
                sirs_score += 1
            wbc = record.get('WBC', 8)
            if wbc > 12 or wbc < 4:
                sirs_score += 1
            scores['sirs'] = sirs_score

            # qSOFA Score
            qsofa_score = 0
            if record.get('Resp', 16) >= 22:
                qsofa_score += 1
            if record.get('SBP', 120) <= 100:
                qsofa_score += 1
            scores['qsofa'] = qsofa_score

            # SOFA Score (partial)
            sofa_score = 0
            platelets = record.get('Platelets', 250)
            if platelets < 20:
                sofa_score += 4
            elif platelets < 50:
                sofa_score += 3
            elif platelets < 100:
                sofa_score += 2
            elif platelets < 150:
                sofa_score += 1

            bilirubin = record.get('Bilirubin_total', 1.0)
            if bilirubin >= 12:
                sofa_score += 4
            elif bilirubin >= 6:
                sofa_score += 3
            elif bilirubin >= 2:
                sofa_score += 2
            elif bilirubin >= 1.2:
                sofa_score += 1
            
            scores['sofa'] = sofa_score
            
        except Exception as e:
            print(f"⚠️ Error calculating clinical scores: {e}")
            scores = {'sirs': 0, 'qsofa': 0, 'sofa': 0}
        
        return scores
    
    def get_comprehensive_prediction(self, patient_data):
        """Get comprehensive prediction combining all models"""
        if not self.is_initialized:
            return None
        
        try:
            # Get baseline predictions
            baseline_preds = self.predict_baseline_models(patient_data)
            
            # Get deep learning predictions
            dl_preds = self.predict_deep_learning_models(patient_data)
            
            # Combine all predictions
            all_predictions = {**baseline_preds, **dl_preds}
            
            # Calculate ensemble prediction
            risk_scores = []
            model_scores = {}
            for name, pred in all_predictions.items():
                if isinstance(pred, dict) and 'risk_score' in pred:
                    score = pred['risk_score']
                    risk_scores.append(score)
                    model_scores[name] = score
            
            # Calculate clinical risk score from clinical scores
            clinical_risk = 0.0
            if 'clinical_scores' in all_predictions:
                scores = all_predictions['clinical_scores']
                # Convert clinical scores to 0-1 risk score
                # SIRS (0-4): weight 0.3, qSOFA (0-3): weight 0.4, SOFA (0-24): weight 0.3
                sirs_norm = min(1.0, scores.get('sirs', 0) / 4.0)
                qsofa_norm = min(1.0, scores.get('qsofa', 0) / 3.0)
                sofa_norm = min(1.0, scores.get('sofa', 0) / 24.0)
                clinical_risk = (sirs_norm * 0.3 + qsofa_norm * 0.4 + sofa_norm * 0.3)
            
            if risk_scores:
                # Ensemble strategy selection
                strategy = str(os.getenv('ENSEMBLE_STRATEGY', 'average')).lower()
                if strategy == 'max':
                    model_combined = float(np.max(risk_scores))
                elif strategy == 'median':
                    model_combined = float(np.median(risk_scores))
                elif strategy == 'weighted':
                    # Parse weights from env: "grud:0.35,lstm:0.25,cnn_lstm:0.2,transformer:0.2"
                    weights_env = os.getenv('ENSEMBLE_WEIGHTS', '')
                    weights: Dict[str, float] = {}
                    try:
                        for item in weights_env.split(','):
                            if ':' in item:
                                k, v = item.split(':', 1)
                                weights[k.strip()] = float(v)
                    except Exception:
                        weights = {}
                    if weights:
                        # Normalize
                        total_w = sum(max(w, 0.0) for w in weights.values()) or 1.0
                        s = 0.0
                        for k, w in weights.items():
                            if k in model_scores:
                                s += model_scores[k] * max(w, 0.0)
                        model_combined = float(s / total_w)
                    else:
                        # Try auto-weights from results (AUPRC-proportional)
                        auto_w = self._load_dl_weights_from_results()
                        if auto_w and len(auto_w) >= 2:
                            print(f"📊 Using auto ensemble weights (AUPRC): {auto_w}")
                            s = 0.0
                            total_w = 0.0
                            for k, w in auto_w.items():
                                if k in model_scores:
                                    s += model_scores[k] * max(w, 0.0)
                                    total_w += max(w, 0.0)
                            model_combined = float(s / (total_w or 1.0))
                        else:
                            # Fallback: equal weights across available models
                            if model_scores:
                                equal_w = 1.0 / float(len(model_scores))
                                model_combined = float(sum(score * equal_w for score in model_scores.values()))
                            else:
                                model_combined = float(np.mean(risk_scores))
                else:
                    # default average
                    model_combined = float(np.mean(risk_scores))
                
                # Conservative fusion with clinical risk if configured
                clinical_priority = str(os.getenv('CLINICAL_PRIORITY', '1')).lower() in ('1', 'true', 'yes')
                ensemble_score = max(model_combined, clinical_risk) if clinical_priority else model_combined
                
                # Calculate agreement score based on variance
                if len(risk_scores) > 1:
                    agreement_score = float(np.clip(1.0 - (np.std(risk_scores) / max(np.mean(risk_scores), 0.01)), 0.0, 1.0))
                else:
                    agreement_score = 0.85
                
                # Ensemble thresholds (configurable via env)
                try:
                    ens_high_t = float(os.getenv('ENSEMBLE_HIGH_THRESHOLD', '0.7'))
                    ens_med_t = float(os.getenv('ENSEMBLE_MEDIUM_THRESHOLD', '0.3'))
                except Exception:
                    ens_high_t, ens_med_t = 0.7, 0.3
                # Use >= instead of > for thresholds
                ensemble_level = 'High' if ensemble_score >= ens_high_t else 'Medium' if ensemble_score >= ens_med_t else 'Low'
            else:
                # Fallback: use clinical risk if available, otherwise default to medium
                ensemble_score = clinical_risk if clinical_risk > 0 else 0.5
                try:
                    ens_high_t = float(os.getenv('ENSEMBLE_HIGH_THRESHOLD', '0.7'))
                    ens_med_t = float(os.getenv('ENSEMBLE_MEDIUM_THRESHOLD', '0.3'))
                except Exception:
                    ens_high_t, ens_med_t = 0.7, 0.3
                ensemble_level = 'High' if ensemble_score >= ens_high_t else 'Medium' if ensemble_score >= ens_med_t else 'Low'
                agreement_score = 0.5
            
            # Debug output (can be removed in production)
            if len(risk_scores) > 0:
                print(f"🔍 DEBUG: Individual risk scores: {[f'{s:.3f}' for s in risk_scores]}")
                print(f"🔍 DEBUG: Model average: {np.mean(risk_scores):.3f}, Clinical risk: {clinical_risk:.3f}")
                print(f"🔍 DEBUG: Final ensemble score: {ensemble_score:.3f} → {ensemble_level}")
            
            return {
                'individual_predictions': all_predictions,
                'ensemble_prediction': {
                    'average_risk_score': float(ensemble_score),
                    'risk_level': ensemble_level,
                    'agreement_score': agreement_score,
                    'model_average': float(np.mean(risk_scores)) if risk_scores else 0.5,
                    'clinical_risk': float(clinical_risk),
                    'recommended_action': self._get_recommendation(ensemble_level)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in comprehensive prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_recommendation(self, risk_level):
        """Get clinical recommendation based on risk level"""
        if risk_level == 'High':
            return 'Immediate clinical attention required'
        elif risk_level == 'Medium':
            return 'Close monitoring recommended'
        else:
            return 'Continue routine monitoring'
    
    def get_feature_importance(self, patient_data):
        """Get feature importance for explainability"""
        try:
            # Simple feature importance based on deviation from normal values
            importance = []
            
            normal_values = {
                'HR': 80, 'Temp': 37, 'Resp': 16, 'WBC': 8, 'SBP': 120,
                'O2Sat': 95, 'Creatinine': 1.0, 'Bilirubin_total': 1.0
            }
            
            for feature, normal_val in normal_values.items():
                if feature in patient_data:
                    value = patient_data[feature]
                    deviation = abs(value - normal_val) / normal_val
                    importance.append({
                        'feature': feature.lower(),
                        'importance': min(deviation, 1.0),
                        'trend': 'increasing' if value > normal_val else 'stable'
                    })
            
            # Sort by importance
            importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return {'top_features': importance[:5]}
            
        except Exception as e:
            print(f"❌ Error calculating feature importance: {e}")
            return {'top_features': []}

# Global instance
integration_system = SepsisPredictionIntegration()

def get_integration_system():
    """Get the global integration system instance"""
    return integration_system

def check_baseline_models_status():
    """Helper function to check if baseline models are loaded and available"""
    system = get_integration_system()
    if not system.is_initialized:
        return {
            'initialized': False,
            'baseline_models_loaded': 0,
            'baseline_models': []
        }
    
    baseline_models = list(system.baseline_models.keys())
    return {
        'initialized': True,
        'baseline_models_loaded': len(baseline_models),
        'baseline_models': baseline_models,
        'deep_learning_models_loaded': len(system.deep_learning_models),
        'deep_learning_models': list(system.deep_learning_models.keys())
    }
