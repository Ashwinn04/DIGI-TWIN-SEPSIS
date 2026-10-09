"""
Training utilities for deep learning models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_score: Current validation score
            model: PyTorch model
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """Training class for deep learning models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []
        
    
class FocalLoss(nn.Module):
    """Binary Focal Loss for class-imbalanced problems."""
    def __init__(self, alpha: Optional[float] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [batch, 1] or [batch]
        # targets: [batch, 1] or [batch]
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * bce_loss
        if self.alpha is not None:
            # Weight positive/negative examples
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def tune_decision_threshold(targets: List[float], preds: List[float], method: str = 'youden') -> Tuple[float, Dict[str, float]]:
    """Compute optimal threshold using Youden J or F1-max.
    Returns (threshold, metrics)."""
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
    y_true = np.array(targets).astype(int)
    y_score = np.array(preds)
    if method == 'youden':
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        idx = np.argmax(j)
        best_thr = thr[idx]
        return float(best_thr), {'tpr': float(tpr[idx]), 'fpr': float(fpr[idx])}
    else:
        precision, recall, thr = precision_recall_curve(y_true, y_score)
        thr = np.append(thr, 1.0)  # align lengths
        f1_vals = (2 * precision * recall) / (precision + recall + 1e-8)
        idx = np.nanargmax(f1_vals)
        best_thr = thr[idx if idx < len(thr) else -1]
        return float(best_thr), {'precision': float(precision[idx]), 'recall': float(recall[idx]), 'f1': float(f1_vals[idx])}

    def train_epoch(self, train_loader, criterion, optimizer, use_amp: bool = True, multitask: bool = False, tto_weight: float = 0.2) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # For calculating AUROC during training
        all_preds = []
        all_targets = []
        
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
        for batch_idx, batch in enumerate(train_loader):
            # Support new dataloader output: features, mask, delta_t, target
            if len(batch) == 5:
                features, masks, delta_t, targets, tto = batch
            elif len(batch) == 4:
                features, masks, delta_t, targets = batch
                tto = None
            else:
                features, masks, targets = batch
                delta_t = None
                tto = None
            features = features.to(self.device)
            masks = masks.to(self.device)
            if delta_t is not None:
                delta_t = delta_t.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
                outputs = self.model(features, masks, delta_t) if delta_t is not None else self.model(features, masks)
                if multitask and isinstance(outputs, tuple):
                    logits, tto_pred = outputs
                else:
                    logits, tto_pred = outputs, None
            
            # Calculate loss
            loss = criterion(logits, targets.float())
            if multitask and tto_pred is not None and tto is not None:
                # time-to-onset regression loss; ignore -1 sentinel
                valid = (tto > -0.5).float()
                if valid.sum() > 0:
                    mse = nn.MSELoss(reduction='none')(tto_pred.view(-1), tto.view(-1))
                    loss = loss + tto_weight * (mse * valid.view(-1)).sum() / valid.sum()
            
            # Backward pass
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            
            # Store predictions for AUROC calculation
            with torch.no_grad():
                preds = torch.sigmoid(logits).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets_cpu.flatten())
        
        avg_loss = total_loss / total_samples
        avg_score = self._calculate_score(all_targets, all_preds)
        
        return avg_loss, avg_score
    
    def validate_epoch(self, val_loader, criterion, multitask: bool = False, tto_weight: float = 0.2) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 5:
                    features, masks, delta_t, targets, tto = batch
                elif len(batch) == 4:
                    features, masks, delta_t, targets = batch
                    tto = None
                else:
                    features, masks, targets = batch
                    delta_t = None
                    tto = None
                features = features.to(self.device)
                masks = masks.to(self.device)
                if delta_t is not None:
                    delta_t = delta_t.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features, masks, delta_t) if delta_t is not None else self.model(features, masks)
                if multitask and isinstance(outputs, tuple):
                    logits, tto_pred = outputs
                else:
                    logits, tto_pred = outputs, None
                loss = criterion(logits, targets.float())
                if multitask and tto_pred is not None and tto is not None:
                    valid = (tto > -0.5).float()
                    if valid.sum() > 0:
                        mse = nn.MSELoss(reduction='none')(tto_pred.view(-1), tto.view(-1))
                        loss = loss + tto_weight * (mse * valid.view(-1)).sum() / valid.sum()
                
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
                preds = torch.sigmoid(logits).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets_cpu.flatten())
        
        avg_loss = total_loss / total_samples
        avg_score = self._calculate_score(all_targets, all_preds)
        
        return avg_loss, avg_score
    
    def _calculate_score(self, targets: List, preds: List) -> float:
        """Calculate AUROC score"""
        from sklearn.metrics import roc_auc_score
        try:
            if len(np.unique(targets)) > 1:
                return roc_auc_score(targets, preds)
            else:
                return 0.5
        except:
            return 0.5
    
    def train(self, train_loader, val_loader, 
              epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-5, patience: int = 10,
              class_weights: Optional[List[float]] = None,
              save_path: Optional[str] = None,
              use_amp: bool = True,
              use_focal_loss: bool = False,
              focal_gamma: float = 2.0,
              multitask: bool = False,
              tto_weight: float = 0.2) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            class_weights: Class weights for imbalanced data
            save_path: Path to save model weights
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                    patience=5)
        
        # Setup loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=None, gamma=focal_gamma)
        elif class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        best_val_score = 0.0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_score = self.train_epoch(train_loader, criterion, optimizer, use_amp=use_amp, multitask=multitask, tto_weight=tto_weight)
            
            # Validate
            val_loss, val_score = self.validate_epoch(val_loader, criterion, multitask=multitask, tto_weight=tto_weight)
            
            # Update learning rate
            scheduler.step(val_score)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_scores.append(train_score)
            self.val_scores.append(val_score)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train AUROC: {train_score:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val AUROC: {val_score:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if early_stopping(val_score, self.model):
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                if save_path:
                    self.save_model(save_path)
        
        # Return training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'best_val_score': best_val_score,
            'epochs_trained': len(self.train_losses)
        }
        
        return history
    
    def save_model(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


def calculate_class_weights(train_loader) -> List[float]:
    """Calculate class weights for imbalanced data"""
    total_samples = 0
    positive_samples = 0
    
    for _, _, targets in train_loader:
        total_samples += len(targets)
        positive_samples += targets.sum().item()
    
    negative_samples = total_samples - positive_samples
    
    if positive_samples == 0 or negative_samples == 0:
        return [1.0, 1.0]
    
    # Return weights as [negative_weight, positive_weight]
    return [total_samples / (2 * negative_samples), 
            total_samples / (2 * positive_samples)]


def save_training_results(results: Dict, model_name: str, 
                         output_dir: str = "outputs/results"):
    """Save training results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_training_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            json_results[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            json_results[key] = float(value)
        else:
            json_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Training results saved to {filepath}")
    return filepath


def run_kfold_cv(model_factory, data, feature_cols: List[str], n_folds: int = 5,
                 batch_size: int = 32, sequence_length: int = 24, prediction_horizon: int = 4,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5, patience: int = 10,
                 use_amp: bool = True, use_focal_loss: bool = False, focal_gamma: float = 2.0,
                 add_feature_engineering: bool = True, rolling_window: int = 3,
                 device: str = 'cpu') -> Dict:
    """
    Run K-fold patient-split cross-validation. model_factory returns a new model each fold.
    """
    from utils.data_loader import get_kfold_patient_splits, create_fold_loaders
    from utils.metrics import calculate_all_metrics
    splits = get_kfold_patient_splits(data, n_folds=n_folds)
    fold_histories = []
    fold_metrics = []
    for fold_idx, (train_patients, val_patients) in enumerate(splits):
        print(f"\n=== Fold {fold_idx+1}/{n_folds} ===")
        train_loader, val_loader = create_fold_loaders(
            data=data,
            train_patients=train_patients,
            val_patients=val_patients,
            batch_size=batch_size,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            features=feature_cols,
            add_feature_engineering=add_feature_engineering,
            rolling_window=rolling_window
        )
        # Infer input size from loader
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[-1]
        model = model_factory(input_size)
        trainer = ModelTrainer(model, device)
        class_weights = calculate_class_weights(train_loader)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=class_weights,
            patience=patience,
            use_amp=use_amp,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma
        )
        fold_histories.append(history)
        # Evaluate on validation with optimal threshold metrics
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    features, masks, delta_t, targets = batch
                else:
                    features, masks, targets = batch
                    delta_t = None
                features = features.to(device)
                masks = masks.to(device)
                if delta_t is not None:
                    delta_t = delta_t.to(device)
                outputs = model(features, masks, delta_t) if delta_t is not None else model(features, masks)
                preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy().flatten())
        metrics = calculate_all_metrics(np.array(all_targets), np.array(all_preds))
        print(f"Fold {fold_idx+1} AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f}")
        fold_metrics.append(metrics)
    # Aggregate
    def agg(name):
        vals = [m[name] for m in fold_metrics if name in m]
        return float(np.mean(vals)), float(np.std(vals))
    summary = {
        'auroc_mean': agg('auroc')[0], 'auroc_std': agg('auroc')[1],
        'auprc_mean': agg('auprc')[0], 'auprc_std': agg('auprc')[1]
    }
    print(f"\nCV Summary: AUROC {summary['auroc_mean']:.4f} ± {summary['auroc_std']:.4f}, "
          f"AUPRC {summary['auprc_mean']:.4f} ± {summary['auprc_std']:.4f}")
    return { 'fold_histories': fold_histories, 'fold_metrics': fold_metrics, 'summary': summary }


def run_training_multiple_seeds(model_factory, input_size: int, seeds: List[int],
                                train_loader, val_loader, device: str = 'cpu',
                                epochs: int = 50, learning_rate: float = 1e-3,
                                weight_decay: float = 1e-5, patience: int = 10,
                                use_amp: bool = True, use_focal_loss: bool = False,
                                focal_gamma: float = 2.0) -> Dict:
    """
    Train the same configuration across multiple random seeds and aggregate results.
    model_factory: function taking input_size -> model instance
    """
    set_random_seeds(0)
    class_weights = calculate_class_weights(train_loader)
    histories = []
    best_scores = []
    for seed in seeds:
        print(f"\n=== Training with seed {seed} ===")
        set_random_seeds(seed)
        model = model_factory(input_size)
        trainer = ModelTrainer(model, device)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=class_weights,
            patience=patience,
            use_amp=use_amp,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma
        )
        histories.append(history)
        best_scores.append(history.get('best_val_score', 0.0))
    summary = {
        'seeds': seeds,
        'best_val_mean': float(np.mean(best_scores)),
        'best_val_std': float(np.std(best_scores))
    }
    print(f"Seeds summary: Val AUROC {summary['best_val_mean']:.4f} ± {summary['best_val_std']:.4f}")
    return { 'histories': histories, 'summary': summary }


def load_training_results(filepath: str) -> Dict:
    """Load training results from JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Convert lists back to numpy arrays where appropriate
    for key, value in results.items():
        if isinstance(value, list) and key in ['train_losses', 'val_losses', 
                                              'train_scores', 'val_scores']:
            results[key] = np.array(value)
    
    return results


def get_device() -> str:
    """Get available device (CPU or GPU)"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
