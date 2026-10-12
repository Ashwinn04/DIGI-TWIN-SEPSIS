"""
Baseline Model Training Script
Trains Logistic Regression, Random Forest, and XGBoost models for sepsis prediction.
Based on Capstone (4).ipynb, updated for macOS paths.
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import time

# Scikit-learn imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss, roc_curve, precision_recall_curve
)

# Feature selection
try:
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    FEATURE_SELECTION_AVAILABLE = True
except Exception:
    FEATURE_SELECTION_AVAILABLE = False

# Imbalance handling (SMOTE)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

# Try to import XGBoost, fallback to HistGradientBoosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, will use HistGradientBoostingClassifier")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def find_dataset_path():
    """Find the dataset CSV file in common locations"""
    possible_paths = [
        'fully_cleaned_sepsis_data.csv',  # Preferred: new cleaned dataset (lowercase)
        'fully_cleaned_Sepsis_data.csv',  # Preferred: new cleaned dataset (capital S)
        'Dataset.csv',
        'Capstone/Dataset.csv',
        '../Dataset.csv',
        '../../Dataset.csv'
    ]
    
    for path in possible_paths:
        full_path = project_root / path
        if full_path.exists():
            return str(full_path)
    
    return None


def compute_clinical_scores(df):
    """Compute clinical scores (SIRS, qSOFA, NEWS2, SOFA)"""
    
    def compute_sirs(row):
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

    def compute_qsofa(row):
        if 'qSOFA_Score_simple' in row.index:
            v = row.get('qSOFA_Score_simple', np.nan)
            if pd.notna(v):
                return int(v)
        rr = row.get('Resp', np.nan)
        sbp = row.get('SBP', np.nan)
        score = 0
        if pd.notna(rr) and rr >= 22:
            score += 1
        if pd.notna(sbp) and sbp <= 100:
            score += 1
        return score

    def compute_news2(row):
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

    def compute_sofa_partial(row):
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

    # Apply functions
    df['SIRS_count'] = df.apply(compute_sirs, axis=1)
    df['SIRS_flag'] = (df['SIRS_count'] >= 2).astype(int)
    df['qSOFA'] = df.apply(compute_qsofa, axis=1)
    df['NEWS2'] = df.apply(compute_news2, axis=1)
    df['SOFA_partial'] = df.apply(compute_sofa_partial, axis=1)
    
    return df


def prepare_features(df):
    """Select and prepare features for training"""
    # Base features
    base_features = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'FiO2', 'pH', 'PaCO2',
        'SaO2', 'BaseExcess', 'HCO3', 'WBC', 'Platelets', 'Creatinine',
        'Bilirubin_total', 'Age'
    ]
    
    # Engineered features (if available)
    engineered_features = [
        'MAP_rolling_mean_6hr', 'MAP_delta_1hr',
        'qSOFA_Resp', 'qSOFA_MAP', 'qSOFA_Score_simple'
    ]
    
    # Clinical scores (computed)
    clinical_scores = ['SIRS_count', 'SIRS_flag', 'qSOFA', 'NEWS2', 'SOFA_partial']
    
    # Combine all features
    all_features = base_features + engineered_features + clinical_scores
    
    # Select only features that exist in the dataframe
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"📋 Using {len(available_features)} features:")
    print(f"   Base: {len([f for f in base_features if f in df.columns])}")
    print(f"   Engineered: {len([f for f in engineered_features if f in df.columns])}")
    print(f"   Clinical scores: {len([f for f in clinical_scores if f in df.columns])}")
    
    return available_features


def find_optimal_threshold(y_true, y_prob, method='youden'):
    """Find optimal threshold using Youden's J or F1-max"""
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = np.argmax(j)
        optimal_threshold = thresholds[idx]
        return optimal_threshold, {'tpr': tpr[idx], 'fpr': fpr[idx], 'j': j[idx]}
    else:  # f1
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        thresholds = np.append(thresholds, 1.0)  # align lengths
        f1_vals = (2 * precision * recall) / (precision + recall + 1e-8)
        idx = np.nanargmax(f1_vals)
        optimal_threshold = thresholds[idx if idx < len(thresholds) else -1]
        return optimal_threshold, {
            'precision': precision[idx], 
            'recall': recall[idx], 
            'f1': f1_vals[idx]
        }


def train_baseline_models(X_train, X_test, y_train, y_test, output_dir='outputs/models'):
    """Train all baseline models with proper validation calibration and class imbalance handling"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Model selection via env var TRAIN_ONLY (comma-separated): e.g., "xgboost" or "lr,rf"
    train_only_env = os.getenv('TRAIN_ONLY', '').strip().lower()
    if train_only_env:
        selected = set(s.strip() for s in train_only_env.split(',') if s.strip())
        # Map common aliases
        alias_map = {
            'lr': 'logistic_regression',
            'logreg': 'logistic_regression',
            'rf': 'random_forest',
            'xgb': 'xgboost',
            'hgb': 'xgboost'
        }
        normalized = set(alias_map.get(s, s) for s in selected)
        train_lr = 'logistic_regression' in normalized
        train_rf = 'random_forest' in normalized
        train_xgb = 'xgboost' in normalized
    else:
        train_lr = True
        train_rf = True
        train_xgb = True
    
    # Imputation and scaling
    print("\n" + "="*60)
    print("Preprocessing")
    print("="*60)
    
    # REUSE_PREPROCESSORS=1: apply the already-deployed imputer/scaler/selector (never re-fit or overwrite
    # them), so a single model can be (re)trained without invalidating the other deployed models.
    reuse = os.getenv('REUSE_PREPROCESSORS', '').strip().lower() in ('1', 'true', 'yes')
    fs_path = os.path.join(output_dir, 'feature_selector.pkl')
    if reuse:
        with open(os.path.join(output_dir, 'baseline_imputer.pkl'), 'rb') as f:
            imputer = pickle.load(f)
        with open(os.path.join(output_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(fs_path, 'rb') as f:
            fs = pickle.load(f)
        X_train_fs = fs.transform(scaler.transform(imputer.transform(X_train)))
        X_test_fs = fs.transform(scaler.transform(imputer.transform(X_test)))
        print(f"✅ Reusing deployed preprocessors from {output_dir} ({X_train_fs.shape[1]} features)")
    else:
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # Feature selection (SelectKBest with mutual information)
        if FEATURE_SELECTION_AVAILABLE:
            try:
                k = min(40, X_train_scaled.shape[1])
                fs = SelectKBest(score_func=mutual_info_classif, k=k)
                X_train_fs = fs.fit_transform(X_train_scaled, y_train)
                X_test_fs = fs.transform(X_test_scaled)
                with open(fs_path, 'wb') as f:
                    pickle.dump(fs, f)
                print(f"✅ Saved feature selector (k={k}) to {fs_path}")
            except Exception as e:
                print(f"⚠️ Feature selection failed ({e}), proceeding without it")
                X_train_fs, X_test_fs = X_train_scaled, X_test_scaled
                fs = None
        else:
            X_train_fs, X_test_fs = X_train_scaled, X_test_scaled
            fs = None

    # Create validation split from training (patient-level already applied upstream if available)
    print("\n" + "="*60)
    print("Creating validation split (for calibration)")
    print("="*60)
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train_fs, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"✅ Train-sub: {X_tr_sub.shape}, Val: {X_val_sub.shape}")

    # Apply SMOTE ONLY for LR on the train-sub set (if LR is selected)
    smote_applied = False
    if train_lr:
        if SMOTE_AVAILABLE:
            try:
                pos = int(y_tr_sub.sum())
                k_neighbors = 1 if pos <= 1 else min(5, pos - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sub, y_tr_sub)
                print(f"✅ Applied SMOTE for LR only: {X_tr_sub.shape} -> {X_tr_bal.shape}")
                smote_applied = True
            except Exception as e:
                print(f"⚠️ SMOTE failed for LR path ({e}), proceeding without it")
                X_tr_bal, y_tr_bal = X_tr_sub, y_tr_sub
        else:
            print("⚠️ imblearn not available; skipping SMOTE")
            X_tr_bal, y_tr_bal = X_tr_sub, y_tr_sub

    # Save preprocessing components (left untouched when reusing the deployed ones)
    if not reuse:
        imputer_path = os.path.join(output_dir, 'baseline_imputer.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')

        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        print(f"✅ Saved imputer to {imputer_path}")

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Saved scaler to {scaler_path}")

    # 1/3. Logistic Regression (with light hyperparameter tuning)
    if train_lr:
        print("\n" + "="*60)
        print("Training Logistic Regression")
        print("="*60)
        
        lr_base = LogisticRegression(
            class_weight=None if smote_applied else 'balanced',
            max_iter=2000,
            random_state=42,
            solver='liblinear'
        )
        try:
            lr_param_dist = {
                'C': np.logspace(-3, 2, 20)
            }
            lr_search = RandomizedSearchCV(
                lr_base,
                param_distributions=lr_param_dist,
                n_iter=10,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=2
            )
            print(f"🔍 LR hyperparameter search: {lr_search.n_iter} candidates × {lr_search.cv} folds = {lr_search.n_iter * lr_search.cv} fits")
            t0 = time.time()
            lr_search.fit(X_tr_bal, y_tr_bal)
            print(f"⏱️ LR search completed in {time.time() - t0:.1f}s")
            lr_best = lr_search.best_estimator_
            print(f"✅ LR best params: {lr_search.best_params_}")
        except Exception as e:
            print(f"⚠️ LR tuning failed ({e}), using defaults")
            lr_best = lr_base
        
        print("📏 Calibrating LR on validation (isotonic, prefit)...")
        t0 = time.time()
        # Fit LR on (possibly SMOTE) train-sub
        lr_best.fit(X_tr_bal, y_tr_bal)
        # Calibrate on true-prevalence validation set
        lr_cal = CalibratedClassifierCV(lr_best, cv='prefit', method='isotonic')
        lr_cal.fit(X_val_sub, y_val_sub)
        print(f"⏱️ LR calibration completed in {time.time() - t0:.1f}s (prefit on validation)")
        
        y_prob_lr = lr_cal.predict_proba(X_test_fs)[:, 1]
        
        # Find optimal threshold (F1-based)
        opt_threshold_lr, _ = find_optimal_threshold(y_test, y_prob_lr, method='f1')
        y_pred_lr = (y_prob_lr >= opt_threshold_lr).astype(int)
        
        # Metrics
        auroc_lr = roc_auc_score(y_test, y_prob_lr)
        auprc_lr = average_precision_score(y_test, y_prob_lr)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
        sensitivity_lr = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_lr = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_lr = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_lr = 2 * precision_lr * sensitivity_lr / (precision_lr + sensitivity_lr) if (precision_lr + sensitivity_lr) > 0 else 0
        brier_lr = brier_score_loss(y_test, y_prob_lr)
        accuracy_lr = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"AUROC: {auroc_lr:.4f}")
        print(f"AUPRC: {auprc_lr:.4f}")
        print(f"Accuracy: {accuracy_lr:.4f}")
        print(f"Optimal Threshold: {opt_threshold_lr:.4f}")
        print(f"Sensitivity (Recall): {sensitivity_lr:.4f}")
        print(f"Specificity: {specificity_lr:.4f}")
        print(f"Precision: {precision_lr:.4f}")
        print(f"F1 Score: {f1_lr:.4f}")
        print(f"Brier Score: {brier_lr:.4f}")
        
        # Save model
        lr_path = os.path.join(output_dir, 'logistic_regression.pkl')
        with open(lr_path, 'wb') as f:
            pickle.dump(lr_cal, f)
        print(f"✅ Saved model to {lr_path}")
        
        results['logistic_regression'] = {
            'auroc': auroc_lr,
            'auprc': auprc_lr,
            'accuracy': accuracy_lr,
            'sensitivity': sensitivity_lr,
            'specificity': specificity_lr,
            'precision': precision_lr,
            'f1': f1_lr,
            'brier': brier_lr,
            'optimal_threshold': float(opt_threshold_lr)
        }

    # 2/3. Random Forest (with light hyperparameter tuning)
    if train_rf:
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        rf_base = RandomForestClassifier(
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        try:
            # Bounded depth and leaf size keep the pickle small enough to ship and load quickly
            # (unbounded depth with single-sample leaves produced a 1.3 GB model).
            rf_param_dist = {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 4, 8]
            }
            rf_search = RandomizedSearchCV(
                rf_base,
                param_distributions=rf_param_dist,
                n_iter=12,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=2
            )
            total_fits = rf_search.n_iter * rf_search.cv
            print(f"🔍 RF hyperparameter search: {rf_search.n_iter} candidates × {rf_search.cv} folds = {total_fits} fits")
            print("⏳ This step can take several minutes depending on CPU cores.")
            t0 = time.time()
            rf_search.fit(X_tr_sub, y_tr_sub)
            print(f"⏱️ RF search completed in {time.time() - t0:.1f}s")
            rf_best = rf_search.best_estimator_
            print(f"✅ RF best params: {rf_search.best_params_}")
        except Exception as e:
            print(f"⚠️ RF tuning failed ({e}), using defaults")
            rf_best = rf_base
        
        print("📏 Calibrating RF (isotonic, cv=5)...")
        t0 = time.time()
        # Fit RF on original (no SMOTE) train-sub
        rf_best.fit(X_tr_sub, y_tr_sub)
        # Calibrate on true-prevalence validation set
        rf_cal = CalibratedClassifierCV(rf_best, cv='prefit', method='isotonic')
        rf_cal.fit(X_val_sub, y_val_sub)
        print(f"⏱️ RF calibration completed in {time.time() - t0:.1f}s (prefit on validation)")
        
        y_prob_rf = rf_cal.predict_proba(X_test_fs)[:, 1]
        
        # Find optimal threshold (F1-based)
        opt_threshold_rf, _ = find_optimal_threshold(y_test, y_prob_rf, method='f1')
        y_pred_rf = (y_prob_rf >= opt_threshold_rf).astype(int)
        
        # Metrics
        auroc_rf = roc_auc_score(y_test, y_prob_rf)
        auprc_rf = average_precision_score(y_test, y_prob_rf)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
        sensitivity_rf = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_rf = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_rf = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_rf = 2 * precision_rf * sensitivity_rf / (precision_rf + sensitivity_rf) if (precision_rf + sensitivity_rf) > 0 else 0
        brier_rf = brier_score_loss(y_test, y_prob_rf)
        accuracy_rf = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"AUROC: {auroc_rf:.4f}")
        print(f"AUPRC: {auprc_rf:.4f}")
        print(f"Accuracy: {accuracy_rf:.4f}")
        print(f"Optimal Threshold: {opt_threshold_rf:.4f}")
        print(f"Sensitivity (Recall): {sensitivity_rf:.4f}")
        print(f"Specificity: {specificity_rf:.4f}")
        print(f"Precision: {precision_rf:.4f}")
        print(f"F1 Score: {f1_rf:.4f}")
        print(f"Brier Score: {brier_rf:.4f}")
        
        # Save model
        rf_path = os.path.join(output_dir, 'random_forest.pkl')
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_cal, f)
        print(f"✅ Saved model to {rf_path}")
        
        results['random_forest'] = {
            'auroc': auroc_rf,
            'auprc': auprc_rf,
            'accuracy': accuracy_rf,
            'sensitivity': sensitivity_rf,
            'specificity': specificity_rf,
            'precision': precision_rf,
            'f1': f1_rf,
            'brier': brier_rf,
            'optimal_threshold': float(opt_threshold_rf)
        }

    # 3/3. XGBoost or HistGradientBoosting (with light hyperparameter tuning)
    if train_xgb:
        print("\n" + "="*60)
        if XGBOOST_AVAILABLE:
            print("Training XGBoost")
        else:
            print("Training HistGradientBoosting (XGBoost fallback)")
        print("="*60)
        
        if XGBOOST_AVAILABLE:
            # Compute scale_pos_weight from original train-sub prevalence
            num_pos = float(y_tr_sub.sum())
            num_neg = float((y_tr_sub == 0).sum())
            spw = num_neg / max(num_pos, 1.0)
            xgb_base = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='aucpr',
                scale_pos_weight=spw,
                random_state=42,
                n_jobs=-1,
                verbosity=1
            )
            try:
                xgb_param_dist = {
                    'n_estimators': [300, 500, 800],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
                xgb_search = RandomizedSearchCV(
                    xgb_base,
                    param_distributions=xgb_param_dist,
                    n_iter=15,
                    scoring='roc_auc',
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    verbose=2
                )
                print(f"🔍 XGB hyperparameter search: {xgb_search.n_iter} candidates × {xgb_search.cv} folds = {xgb_search.n_iter * xgb_search.cv} fits")
                t0 = time.time()
                xgb_search.fit(X_tr_sub, y_tr_sub)
                print(f"⏱️ XGB search completed in {time.time() - t0:.1f}s")
                xgb_best = xgb_search.best_estimator_
                print(f"✅ XGB best params: {xgb_search.best_params_}")
            except Exception as e:
                print(f"⚠️ XGB tuning failed ({e}), using defaults")
                xgb_best = xgb_base
        else:
            hgb_base = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
            try:
                hgb_param_dist = {
                    'max_iter': [200, 300, 400],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8]
                }
                hgb_search = RandomizedSearchCV(
                    hgb_base,
                    param_distributions=hgb_param_dist,
                    n_iter=10,
                    scoring='roc_auc',
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    verbose=2
                )
                print(f"🔍 HGB hyperparameter search: {hgb_search.n_iter} candidates × {hgb_search.cv} folds = {hgb_search.n_iter * hgb_search.cv} fits")
                t0 = time.time()
                hgb_search.fit(X_tr_sub, y_tr_sub)
                print(f"⏱️ HGB search completed in {time.time() - t0:.1f}s")
                xgb_best = hgb_search.best_estimator_
                print(f"✅ HGB best params: {hgb_search.best_params_}")
            except Exception as e:
                print(f"⚠️ HGB tuning failed ({e}), using defaults")
                xgb_best = hgb_base
        
        print("📏 Calibrating XGB/HGB (isotonic, cv=5)...")
        t0 = time.time()
        # Refit best XGB with early stopping on validation (true prevalence)
        try:
            xgb_best.set_params(eval_metric='aucpr')
        except Exception:
            pass
        # Use callback-based early stopping for broad version compatibility
        try:
            es_cb = [xgb.callback.EarlyStopping(
                rounds=50, save_best=True, maximize=True, data_name='validation_0', metric_name='aucpr'
            )]
        except Exception:
            es_cb = None
        xgb_best.fit(
            X_tr_sub, y_tr_sub,
            eval_set=[(X_val_sub, y_val_sub)],
            verbose=False,
            callbacks=es_cb
        )
        # Calibrate on validation set (prefit)
        xgb_cal = CalibratedClassifierCV(xgb_best, cv='prefit', method='isotonic')
        xgb_cal.fit(X_val_sub, y_val_sub)
        print(f"⏱️ XGB/HGB calibration completed in {time.time() - t0:.1f}s (prefit on validation)")
        
        y_prob_xgb = xgb_cal.predict_proba(X_test_fs)[:, 1]
        
        # Find optimal threshold (F1-based)
        opt_threshold_xgb, _ = find_optimal_threshold(y_test, y_prob_xgb, method='f1')
        y_pred_xgb = (y_prob_xgb >= opt_threshold_xgb).astype(int)
        
        # Metrics
        auroc_xgb = roc_auc_score(y_test, y_prob_xgb)
        auprc_xgb = average_precision_score(y_test, y_prob_xgb)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
        sensitivity_xgb = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity_xgb = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_xgb = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_xgb = 2 * precision_xgb * sensitivity_xgb / (precision_xgb + sensitivity_xgb) if (precision_xgb + sensitivity_xgb) > 0 else 0
        brier_xgb = brier_score_loss(y_test, y_prob_xgb)
        accuracy_xgb = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"AUROC: {auroc_xgb:.4f}")
        print(f"AUPRC: {auprc_xgb:.4f}")
        print(f"Accuracy: {accuracy_xgb:.4f}")
        print(f"Optimal Threshold: {opt_threshold_xgb:.4f}")
        print(f"Sensitivity (Recall): {sensitivity_xgb:.4f}")
        print(f"Specificity: {specificity_xgb:.4f}")
        print(f"Precision: {precision_xgb:.4f}")
        print(f"F1 Score: {f1_xgb:.4f}")
        print(f"Brier Score: {brier_xgb:.4f}")
        
        # Save model
        xgb_path = os.path.join(output_dir, 'xgboost.pkl')
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_cal, f)
        print(f"✅ Saved model to {xgb_path}")
        
        results['xgboost'] = {
            'auroc': auroc_xgb,
            'auprc': auprc_xgb,
            'accuracy': accuracy_xgb,
            'sensitivity': sensitivity_xgb,
            'specificity': specificity_xgb,
            'precision': precision_xgb,
            'f1': f1_xgb,
            'brier': brier_xgb,
            'optimal_threshold': float(opt_threshold_xgb)
        }

    return results


def main():
    """Main training function"""
    print("="*60)
    print("Baseline Model Training for Sepsis Prediction")
    print("="*60)
    
    # Find dataset
    dataset_path = find_dataset_path()
    if dataset_path is None:
        print("❌ Error: Could not find Dataset.csv")
        print("   Searched in:")
        for path in ['Dataset.csv', 'Capstone/Dataset.csv']:
            print(f"     - {project_root / path}")
        return
    
    print(f"\n📂 Loading dataset from: {dataset_path}")
    
    # Load data
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check for target column
    target_col = None
    for col in ['SepsisLabel', 'Sepsis_Label']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print("❌ Error: Could not find SepsisLabel or Sepsis_Label column")
        return
    
    print(f"✅ Target column: {target_col}")
    print(f"   Positive cases: {df[target_col].sum()} ({df[target_col].mean()*100:.2f}%)")
    
    # Compute clinical scores
    print("\n📊 Computing clinical scores...")
    df = compute_clinical_scores(df)
    
    # Prepare features
    feature_cols = prepare_features(df)
    if os.getenv('REUSE_PREPROCESSORS', '').strip().lower() in ('1', 'true', 'yes'):
        # the deployed imputer defines the feature set and order the other models were trained on
        with open(os.path.join('outputs', 'models', 'baseline_imputer.pkl'), 'rb') as f:
            deployed = list(pickle.load(f).feature_names_in_)
        missing = [c for c in deployed if c not in df.columns]
        if missing:
            print(f"❌ Dataset lacks features the deployed preprocessors expect: {missing}")
            return
        feature_cols = deployed
        print(f"✅ Using the {len(feature_cols)} features of the deployed preprocessors")
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    
    print(f"\n📋 Feature matrix shape: {X.shape}")
    print(f"   Missing values: {X.isna().sum().sum()} ({X.isna().sum().sum() / X.size * 100:.2f}%)")
    
    # Train-test split (patient-level if Patient_ID exists)
    print("\n🔀 Splitting data...")
    if 'Patient_ID' in df.columns:
        unique_pats = df['Patient_ID'].unique()
        train_pats, test_pats = train_test_split(
            unique_pats, test_size=0.2, random_state=42
        )
        train_idx = df['Patient_ID'].isin(train_pats)
        test_idx = df['Patient_ID'].isin(test_pats)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        print(f"✅ Patient-level split:")
        print(f"   Train: {len(train_pats)} patients, {len(X_train)} records")
        print(f"   Test: {len(test_pats)} patients, {len(X_test)} records")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"✅ Random stratified split:")
        print(f"   Train: {len(X_train)} records")
        print(f"   Test: {len(X_test)} records")
    
    print(f"\n   Train positive rate: {y_train.mean()*100:.2f}%")
    print(f"   Test positive rate: {y_test.mean()*100:.2f}%")
    
    # Train models
    results = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Save results
    results_path = os.path.join('outputs', 'models', 'baseline_models_metrics.json')
    if os.path.exists(results_path) and os.getenv('TRAIN_ONLY', '').strip():
        # partial retrain: keep the other models' metrics
        with open(results_path) as f:
            merged = json.load(f)
        merged.update(results)
        results = merged
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved results to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"{'Model':<25} {'AUROC':<10} {'AUPRC':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1':<10} {'Threshold':<10}")
    print("-"*90)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['auroc']:<10.4f} {metrics['auprc']:<10.4f} "
              f"{metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f} "
              f"{metrics['f1']:<10.4f} {metrics['optimal_threshold']:<10.4f}")
    
    print("\n✅ Training completed!")
    print(f"📁 Models saved to: outputs/models/")


if __name__ == '__main__':
    main()

