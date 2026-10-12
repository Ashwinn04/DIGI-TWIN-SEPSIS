"""
Data loading utilities for ICU time-series data
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import warnings
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings('ignore')


def create_patient_splits(data: pd.DataFrame, output_dir: str = 'outputs/cache',
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          seed: int = 42) -> Dict[str, List[int]]:
    """
    Create and save patient-level splits for train, validation, and test sets.
    If split files already exist, load them.
    """
    os.makedirs(output_dir, exist_ok=True)
    split_file = os.path.join(output_dir, f'patient_splits_seed{seed}.json')

    if os.path.exists(split_file):
        print(f"Loading existing patient splits from {split_file}")
        with open(split_file, 'r') as f:
            patient_splits = json.load(f)
        return patient_splits

    print("Creating new patient splits...")
    unique_patients = data['Patient_ID'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)

    n_train = int(len(unique_patients) * train_ratio)
    n_val = int(len(unique_patients) * val_ratio)

    train_patients = unique_patients[:n_train].tolist()
    val_patients = unique_patients[n_train:n_train + n_val].tolist()
    test_patients = unique_patients[n_train + n_val:].tolist()

    patient_splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }

    with open(split_file, 'w') as f:
        json.dump(patient_splits, f)

    print(f"Saved patient splits to {split_file}")
    return patient_splits


def fit_and_save_imputer_scaler(train_data: pd.DataFrame, feature_cols: List[str],
                                output_dir: str = 'outputs/cache'):
    """Fit imputer and scaler on training data and save them."""
    os.makedirs(output_dir, exist_ok=True)
    imputer_path = os.path.join(output_dir, 'imputer.joblib')
    scaler_path = os.path.join(output_dir, 'scaler.joblib')

    # Fit imputer (e.g., mean imputation)
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(train_data[feature_cols])
    joblib.dump(imputer, imputer_path)
    print(f"Fitted and saved imputer to {imputer_path}")

    # Fit scaler
    scaler = StandardScaler()
    # Apply imputer before fitting scaler
    train_imputed = pd.DataFrame(imputer.transform(train_data[feature_cols]), columns=feature_cols)
    scaler.fit(train_imputed)
    joblib.dump(scaler, scaler_path)
    print(f"Fitted and saved scaler to {scaler_path}")
    
    return imputer, scaler


def apply_imputation_and_scaling(data: pd.DataFrame, feature_cols: List[str],
                                 imputer: SimpleImputer, scaler: StandardScaler) -> pd.DataFrame:
    """Apply pre-fitted imputer and scaler to the data."""
    data_copy = data.copy()
    
    # Keep non-feature columns
    metadata_cols = [col for col in data.columns if col not in feature_cols]
    metadata_df = data_copy[metadata_cols]
    
    # Apply imputation and scaling
    features_imputed = imputer.transform(data_copy[feature_cols])
    features_scaled = scaler.transform(features_imputed)
    
    features_df = pd.DataFrame(features_scaled, columns=feature_cols, index=data_copy.index)
    
    return pd.concat([metadata_df, features_df], axis=1)


class ICUDataset(Dataset):
    """
    PyTorch Dataset for ICU time-series data with variable-length sequences
    """
    
    def __init__(self, data: pd.DataFrame, features: List[str],
                 sequence_length: int = 48, 
                 prediction_horizon: int = 6,
                 step_size: int = 1):
        """
        Initialize ICU Dataset
        
        Args:
            data: DataFrame with columns ['Patient_ID', 'Time', 'Sepsis_Label', ...features]
            features: List of feature column names
            sequence_length: Maximum sequence length in hours
            prediction_horizon: Hours ahead to predict sepsis
            step_size: Step size for creating sequences in hours
        """
        self.data = data.copy()
        self.features = features
        self.n_features = len(self.features)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.step_size = step_size
            
        # Group by patient and create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict]:
        """Create sequences grouped by patient using a sliding window."""
        sequences = []
        patient_groups = self.data.groupby('Patient_ID')
        
        for patient_id, patient_data in patient_groups:
            patient_data = patient_data.sort_values('Time').reset_index(drop=True)
            
            # Find the max hour for this patient
            max_hour = int(patient_data['Time'].max())
            
            # Create a complete time grid for this patient
            time_grid = pd.DataFrame({'Time': range(max_hour + 1)})
            patient_data = pd.merge(time_grid, patient_data, on='Time', how='left')
            patient_data['Patient_ID'] = patient_id # Fill patient ID
            
            # Forward-fill Sepsis_Label, then features
            patient_data['Sepsis_Label'] = patient_data['Sepsis_Label'].ffill()
            
            # Note: imputation and scaling should be done before this
            
            for i in range(0, len(patient_data) - self.sequence_length - self.prediction_horizon, self.step_size):
                
                seq_end_idx = i + self.sequence_length
                
                # Input sequence
                seq_data = patient_data.iloc[i:seq_end_idx]
                
                # Target: sepsis label at prediction_horizon hours ahead
                target_idx = seq_end_idx + self.prediction_horizon - 1
                
                if target_idx >= len(patient_data):
                    continue

                target_label = patient_data.iloc[target_idx]['Sepsis_Label']
                
                # Extract features and handle potential NaNs from merge
                features_matrix = seq_data[self.features].values
                mask_matrix = ~np.isnan(features_matrix)
                
                # Fill NaN values with 0 (should be minimal after imputation)
                features_matrix = np.nan_to_num(features_matrix, nan=0.0)

                # Time delta
                time_stamps = seq_data['Time'].values
                delta_t = np.diff(time_stamps, prepend=time_stamps[0]).reshape(-1, 1)

                sequences.append({
                    'features': features_matrix,
                    'mask': mask_matrix,
                    'delta_t': delta_t,
                    'target': target_label,
                    'patient_id': patient_id,
                    'time_start': patient_data.iloc[i]['Time'],
                    'time_end': patient_data.iloc[seq_end_idx - 1]['Time']
                })
                
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence"""
        seq = self.sequences[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(seq['features'])
        mask = torch.BoolTensor(seq['mask'])
        delta_t = torch.FloatTensor(seq['delta_t'])
        target = torch.FloatTensor([seq['target']]) # Use FloatTensor for BCEWithLogitsLoss
        
        return features, mask, delta_t, target


def create_data_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                       test_data: pd.DataFrame, features: List[str],
                       batch_size: int = 64, sequence_length: int = 48, 
                       prediction_horizon: int = 6, step_size: int = 1
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        train_data: Training data
        val_data: Validation data  
        test_data: Test data
        features: Feature column names
        batch_size: Batch size for training
        sequence_length: Maximum sequence length in hours
        prediction_horizon: Prediction horizon in hours
        step_size: Step size in hours for creating sequences
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = ICUDataset(train_data, features, sequence_length, prediction_horizon, step_size)
    val_dataset = ICUDataset(val_data, features, sequence_length, prediction_horizon, step_size)
    test_dataset = ICUDataset(test_data, features, sequence_length, prediction_horizon, step_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                           num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def load_preprocessed_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load preprocessed data from Person A
    
    Args:
        data_path: Path to the preprocessed dataset
        
    Returns:
        Tuple of (data, feature_columns)
    """
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Handle different column name formats
        # Rename columns to standard format if needed
        column_mapping = {}
        if 'Hour' in data.columns and 'Time' not in data.columns:
            column_mapping['Hour'] = 'Time'
        if 'SepsisLabel' in data.columns and 'Sepsis_Label' not in data.columns:
            column_mapping['SepsisLabel'] = 'Sepsis_Label'
        
        if column_mapping:
            data = data.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")
        
        # Get feature columns (exclude metadata columns)
        metadata_cols = ['Patient_ID', 'Time', 'Sepsis_Label', 'Unnamed: 0']
        feature_cols = [col for col in data.columns if col not in metadata_cols]
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Unique patients: {data['Patient_ID'].nunique()}")
        print(f"Class distribution:")
        print(data['Sepsis_Label'].value_counts())
        print(f"Percentage:")
        print(data['Sepsis_Label'].value_counts(normalize=True) * 100)
        
        return data, feature_cols
        
    except FileNotFoundError:
        print(f"Data file not found at {data_path}")
        print("Creating sample data for testing...")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        print("Creating sample data for testing...")
        return create_sample_data()


def create_sample_data(n_patients: int = 100, n_hours: int = 72, 
                     n_features: int = 40) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create sample data for testing when real data is not available
    
    Args:
        n_patients: Number of patients
        n_hours: Hours per patient
        n_features: Number of features
        
    Returns:
        Tuple of (sample_data, feature_columns)
    """
    np.random.seed(42)
    
    data = []
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    for patient_id in range(n_patients):
        # Random sepsis onset time (if any)
        sepsis_onset = np.random.choice([None, np.random.randint(24, n_hours-12)])
        
        for hour in range(n_hours):
            # Generate features with some missing values
            features = np.random.normal(0, 1, n_features)
            
            # Add missing values (15% missing)
            missing_mask = np.random.random(n_features) < 0.15
            features[missing_mask] = np.nan
            
            # Sepsis label
            sepsis_label = 0
            if sepsis_onset is not None and hour >= sepsis_onset:
                sepsis_label = 1
            
            row = {
                'Patient_ID': patient_id,
                'Time': hour,
                'Sepsis_Label': sepsis_label,
                **{col: val for col, val in zip(feature_cols, features)}
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Created sample data: {df.shape}")
    print(f"Class distribution: {df['Sepsis_Label'].value_counts()}")
    
    return df, feature_cols


def get_train_val_test_split(data: pd.DataFrame,
                             patient_splits: Optional[Dict[str, List[int]]] = None,
                             train_ratio: float = 0.7, val_ratio: float = 0.15,
                             seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by patient ID (never within a patient, to avoid leakage).

    Args:
        data: Full dataset
        patient_splits: Dict of patient-ID lists for 'train'/'val'/'test' (see create_patient_splits).
            If omitted, the same seeded patient shuffle as create_patient_splits is used.
        train_ratio, val_ratio, seed: only used when patient_splits is omitted.
    """
    if patient_splits is None:
        patients = data['Patient_ID'].unique()
        np.random.default_rng(seed).shuffle(patients)
        n_train = int(len(patients) * train_ratio)
        n_val = int(len(patients) * val_ratio)
        patient_splits = {'train': patients[:n_train].tolist(),
                          'val': patients[n_train:n_train + n_val].tolist(),
                          'test': patients[n_train + n_val:].tolist()}

    train_patients = patient_splits['train']
    val_patients = patient_splits['val']
    test_patients = patient_splits['test']
    
    train_data = data[data['Patient_ID'].isin(train_patients)].copy()
    val_data = data[data['Patient_ID'].isin(val_patients)].copy()
    test_data = data[data['Patient_ID'].isin(test_patients)].copy()
    
    print(f"Train: {len(train_patients)} patients, {len(train_data)} records")
    print(f"Val: {len(val_patients)} patients, {len(val_data)} records")
    print(f"Test: {len(test_patients)} patients, {len(test_data)} records")
    
    return train_data, val_data, test_data
