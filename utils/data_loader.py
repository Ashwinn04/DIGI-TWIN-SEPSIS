"""
Data loading utilities for ICU time-series data
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ICUDataset(Dataset):
    """
    PyTorch Dataset for ICU time-series data with variable-length sequences
    """
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 24, 
                 prediction_horizon: int = 4, features: List[str] = None,
                 add_feature_engineering: bool = False, rolling_window: int = 3):
        """
        Initialize ICU Dataset
        
        Args:
            data: DataFrame with columns ['Patient_ID', 'Time', 'Sepsis_Label', ...features]
            sequence_length: Maximum sequence length for padding
            prediction_horizon: Hours ahead to predict (4 or 6)
            features: List of feature column names
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.add_feature_engineering = add_feature_engineering
        self.rolling_window = max(1, int(rolling_window))
        
        # Get feature columns (exclude Patient_ID, Time, Sepsis_Label)
        if features is None:
            self.features = [col for col in data.columns 
                           if col not in ['Patient_ID', 'Time', 'Sepsis_Label']]
        else:
            self.features = features
            
        self.n_features = len(self.features)
        
        # Group by patient and create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict]:
        """Create sequences grouped by patient"""
        sequences = []
        
        for patient_id, patient_data in self.data.groupby('Patient_ID'):
            # Sort by time
            patient_data = patient_data.sort_values('Time').reset_index(drop=True)
            
            # Create sequences with sliding window
            for i in range(len(patient_data) - self.sequence_length):
                # Input sequence
                seq_data = patient_data.iloc[i:i + self.sequence_length]
                
                # Target: sepsis label at prediction_horizon hours ahead
                target_idx = min(i + self.sequence_length + self.prediction_horizon - 1, 
                               len(patient_data) - 1)
                target_label = patient_data.iloc[target_idx]['Sepsis_Label']
                
                # Extract features and create masks for missing values
                features_matrix = seq_data[self.features].values
                mask_matrix = ~np.isnan(features_matrix)

                # Compute per-feature time since last observation (delta_t)
                # Initialize with zeros for the first timestep
                times = seq_data['Time'].values.astype(float)
                delta_t_matrix = np.zeros_like(features_matrix, dtype=float)
                # Track last observed time per feature; initialize to current time so first delta is 0
                last_observed_time = np.full((self.n_features,), times[0], dtype=float)
                for t in range(self.sequence_length):
                    if t > 0:
                        # Increase delta by elapsed time since previous row
                        elapsed = float(times[t] - times[t - 1])
                        # For features not observed at current step, accumulate elapsed time
                        delta_t_matrix[t] = delta_t_matrix[t - 1] + elapsed
                    # For features observed at current step, reset delta to 0
                    observed_mask = mask_matrix[t]
                    delta_t_matrix[t][observed_mask] = 0.0
                
                # Optional: engineer features (deltas, rolling means, shock index)
                engineered_matrix_list = []
                engineered_mask_list = []
                if self.add_feature_engineering:
                    # Compute deltas per feature across time
                    deltas = np.zeros_like(features_matrix, dtype=float)
                    deltas[1:] = np.diff(features_matrix, axis=0)
                    delta_mask = np.zeros_like(mask_matrix, dtype=bool)
                    delta_mask[1:] = mask_matrix[1:] & mask_matrix[:-1]
                    engineered_matrix_list.append(deltas)
                    engineered_mask_list.append(delta_mask)

                    # Rolling means per feature across time
                    roll = np.zeros_like(features_matrix, dtype=float)
                    roll_mask = np.zeros_like(mask_matrix, dtype=bool)
                    w = self.rolling_window
                    if w > 1:
                        # Use cumulative sums with NaN handling via masks
                        filled = np.nan_to_num(features_matrix, nan=0.0)
                        counts = mask_matrix.astype(int)
                        csum = np.cumsum(filled, axis=0)
                        ccount = np.cumsum(counts, axis=0)
                        # pad with leading zeros for correct window subtraction
                        csum0 = np.vstack([np.zeros((1, csum.shape[1])), csum])
                        ccount0 = np.vstack([np.zeros((1, ccount.shape[1])), ccount])
                        window_sum = csum0[w:] - csum0[:-w]  # [seq_len - w + 1, n_features]
                        window_count = ccount0[w:] - ccount0[:-w]
                        # place results starting at index w-1
                        roll[w-1:] = window_sum / np.where(window_count == 0, 1, window_count)
                        roll_mask[w-1:] = window_count == w
                    else:
                        roll = np.nan_to_num(features_matrix, nan=0.0)
                        roll_mask = mask_matrix.copy()
                    engineered_matrix_list.append(roll)
                    engineered_mask_list.append(roll_mask)

                    # Shock index if HR and SBP columns exist
                    hr_candidates = [c for c in self.features if c.lower() in ['hr', 'heartrate', 'heart_rate']]
                    sbp_candidates = [c for c in self.features if c.lower() in ['sbp', 'sysbp', 'systolicbp', 'systolic_bp']]
                    if hr_candidates and sbp_candidates:
                        hr_col = hr_candidates[0]
                        sbp_col = sbp_candidates[0]
                        hr_vals = seq_data[hr_col].values.astype(float)
                        sbp_vals = seq_data[sbp_col].values.astype(float)
                        shock = np.divide(hr_vals, sbp_vals, out=np.zeros_like(hr_vals, dtype=float), where=~np.isnan(hr_vals) & ~np.isnan(sbp_vals))
                        shock = shock.reshape(-1, 1)
                        shock_mask = (~np.isnan(hr_vals) & ~np.isnan(sbp_vals)).reshape(-1, 1)
                        engineered_matrix_list.append(shock)
                        engineered_mask_list.append(shock_mask)

                # Concatenate engineered features if any
                if engineered_matrix_list:
                    features_matrix = np.concatenate([features_matrix] + engineered_matrix_list, axis=1)
                    mask_matrix = np.concatenate([mask_matrix] + engineered_mask_list, axis=1)
                    # Update feature count to reflect engineered features
                    # Note: this only affects this sequence; models should infer input_size from data
                
                # Fill NaN values with 0 (will be masked)
                features_matrix = np.nan_to_num(features_matrix, nan=0.0)
                
                # Compute time-to-onset label for multi-task (if future onset exists)
                # Find first onset time at/after current window end
                remaining = patient_data.iloc[i + self.sequence_length:]
                onset_rows = remaining[remaining['Sepsis_Label'] == 1]
                if len(onset_rows) > 0:
                    onset_time = float(onset_rows.iloc[0]['Time'])
                    time_to_onset = max(0.0, onset_time - float(seq_data['Time'].values[-1]))
                else:
                    time_to_onset = np.nan

                sequences.append({
                    'features': features_matrix,
                    'mask': mask_matrix,
                    'delta_t': delta_t_matrix,
                    'target': target_label,
                    'time_to_onset': time_to_onset,
                    'patient_id': patient_id,
                    'time_start': patient_data.iloc[i]['Time'],
                    'time_end': patient_data.iloc[i + self.sequence_length - 1]['Time'],
                    'times': seq_data['Time'].values.astype(float)
                })
                
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence"""
        seq = self.sequences[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(seq['features'])
        mask = torch.BoolTensor(seq['mask'])
        target = torch.LongTensor([seq['target']])
        delta_t = torch.FloatTensor(seq['delta_t'])
        tto = torch.FloatTensor([seq['time_to_onset'] if not np.isnan(seq['time_to_onset']) else -1.0])
        
        return features, mask, delta_t, target, tto


def create_data_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                       test_data: pd.DataFrame, batch_size: int = 32,
                       sequence_length: int = 24, prediction_horizon: int = 4,
                       features: List[str] = None,
                       add_feature_engineering: bool = False, rolling_window: int = 3) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        train_data: Training data
        val_data: Validation data  
        test_data: Test data
        batch_size: Batch size for training
        sequence_length: Maximum sequence length
        prediction_horizon: Prediction horizon in hours
        features: Feature column names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = ICUDataset(train_data, sequence_length, prediction_horizon, features,
                               add_feature_engineering=add_feature_engineering,
                               rolling_window=rolling_window)
    val_dataset = ICUDataset(val_data, sequence_length, prediction_horizon, features,
                             add_feature_engineering=add_feature_engineering,
                             rolling_window=rolling_window)
    test_dataset = ICUDataset(test_data, sequence_length, prediction_horizon, features,
                              add_feature_engineering=add_feature_engineering,
                              rolling_window=rolling_window)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, test_loader


def create_multi_horizon_loaders(train_data: pd.DataFrame, val_data: pd.DataFrame,
                               test_data: pd.DataFrame, horizons: List[int] = [3, 6, 12],
                               batch_size: int = 32, sequence_length: int = 24,
                               features: List[str] = None,
                               add_feature_engineering: bool = False, rolling_window: int = 3) -> Dict[int, Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Create data loaders for multiple prediction horizons (e.g., 3h, 6h, 12h).
    Returns a dict mapping horizon -> (train_loader, val_loader, test_loader).
    """
    loaders = {}
    for h in horizons:
        loaders[h] = create_data_loaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            batch_size=batch_size,
            sequence_length=sequence_length,
            prediction_horizon=h,
            features=features,
            add_feature_engineering=add_feature_engineering,
            rolling_window=rolling_window
        )
    return loaders


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
        
        # Get feature columns (exclude metadata columns)
        metadata_cols = ['Patient_ID', 'Time', 'Sepsis_Label']
        feature_cols = [col for col in data.columns if col not in metadata_cols]
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Unique patients: {data['Patient_ID'].nunique()}")
        print(f"Class distribution: {data['Sepsis_Label'].value_counts()}")
        
        return data, feature_cols
        
    except FileNotFoundError:
        print(f"Data file not found at {data_path}")
        print("Creating sample data for testing...")
        return create_sample_data()


def create_sample_data(n_patients: int = 100, n_hours: int = 48, 
                     n_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
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
        sepsis_onset = np.random.choice([None, np.random.randint(12, n_hours-6)])
        
        for hour in range(n_hours):
            # Generate features with some missing values
            features = np.random.normal(0, 1, n_features)
            
            # Add missing values (10% missing)
            missing_mask = np.random.random(n_features) < 0.1
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
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by patient ID to avoid data leakage
    
    Args:
        data: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    unique_patients = data['Patient_ID'].unique()
    np.random.shuffle(unique_patients)
    
    n_train = int(len(unique_patients) * train_ratio)
    n_val = int(len(unique_patients) * val_ratio)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]
    
    train_data = data[data['Patient_ID'].isin(train_patients)].copy()
    val_data = data[data['Patient_ID'].isin(val_patients)].copy()
    test_data = data[data['Patient_ID'].isin(test_patients)].copy()
    
    print(f"Train: {len(train_patients)} patients, {len(train_data)} records")
    print(f"Val: {len(val_patients)} patients, {len(val_data)} records")
    print(f"Test: {len(test_patients)} patients, {len(test_data)} records")
    
    return train_data, val_data, test_data


def get_kfold_patient_splits(data: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create patient-level K-fold splits. Returns list of (train_patients, val_patients).
    """
    rng = np.random.RandomState(seed)
    unique_patients = data['Patient_ID'].unique()
    rng.shuffle(unique_patients)
    folds = np.array_split(unique_patients, n_folds)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val_patients = folds[i]
        train_patients = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        splits.append((train_patients, val_patients))
    return splits


def create_fold_loaders(data: pd.DataFrame, train_patients: np.ndarray, val_patients: np.ndarray,
                        batch_size: int = 32, sequence_length: int = 24, prediction_horizon: int = 4,
                        features: List[str] = None,
                        add_feature_engineering: bool = False, rolling_window: int = 3) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val loaders for a single fold using provided patient ID arrays.
    """
    train_data = data[data['Patient_ID'].isin(train_patients)].copy()
    val_data = data[data['Patient_ID'].isin(val_patients)].copy()
    train_dataset = ICUDataset(train_data, sequence_length, prediction_horizon, features,
                               add_feature_engineering=add_feature_engineering,
                               rolling_window=rolling_window)
    val_dataset = ICUDataset(val_data, sequence_length, prediction_horizon, features,
                             add_feature_engineering=add_feature_engineering,
                             rolling_window=rolling_window)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


# ========================= External Dataset Adapters ========================= #
def load_physionet2019_sepsis(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load PhysioNet 2019 sepsis data and adapt columns to our schema.
    Expects columns including 'Patient_ID' (or 'patient_id'), 'Hour'/'Time', 'SepsisLabel'/'Sepsis_Label'.
    Returns (dataframe, feature_cols).
    """
    df = pd.read_csv(csv_path)
    rename_map = {}
    if 'patient_id' in df.columns and 'Patient_ID' not in df.columns:
        rename_map['patient_id'] = 'Patient_ID'
    if 'Hour' in df.columns and 'Time' not in df.columns:
        rename_map['Hour'] = 'Time'
    if 'SepsisLabel' in df.columns and 'Sepsis_Label' not in df.columns:
        rename_map['SepsisLabel'] = 'Sepsis_Label'
    df = df.rename(columns=rename_map)
    # Basic validation
    required = {'Patient_ID', 'Time', 'Sepsis_Label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in PhysioNet data: {missing}")
    metadata_cols = ['Patient_ID', 'Time', 'Sepsis_Label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    return df, feature_cols


def load_eicu(csv_path: str, patient_id_col: str = 'Patient_ID', time_col: str = 'Time', label_col: str = 'Sepsis_Label') -> Tuple[pd.DataFrame, List[str]]:
    """
    Load eICU-derived CSV and adapt to our schema using provided column names.
    Returns (dataframe, feature_cols).
    """
    df = pd.read_csv(csv_path)
    rename_map = {}
    if patient_id_col != 'Patient_ID':
        rename_map[patient_id_col] = 'Patient_ID'
    if time_col != 'Time':
        rename_map[time_col] = 'Time'
    if label_col != 'Sepsis_Label':
        rename_map[label_col] = 'Sepsis_Label'
    if rename_map:
        df = df.rename(columns=rename_map)
    required = {'Patient_ID', 'Time', 'Sepsis_Label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in eICU data: {missing}")
    metadata_cols = ['Patient_ID', 'Time', 'Sepsis_Label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    return df, feature_cols
