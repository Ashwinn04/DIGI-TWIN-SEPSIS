"""
Training utilities for deep learning models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import warnings

from .metrics import calculate_all_metrics

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced datasets."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        probas = torch.sigmoid(inputs)
        
        # Calculate focal loss component
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)
        
        loss = focal_weight * bce_loss
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        
        # Apply pos_weight
        if self.pos_weight is not None:
            pos_weight_t = self.pos_weight * targets + (1 - targets)
            loss = pos_weight_t * loss
            
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss."""
    def __init__(self, smoothing: float = 0.05, pos_weight: Optional[torch.Tensor] = None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    def forward(self, inputs, targets):
        with torch.no_grad():
            smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce_loss(inputs, smooth_targets)


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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    print("Restoring best model weights.")
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
        
    def train_epoch(self, train_loader, criterion, optimizer, scaler, grad_clip_value: float = 1.0) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # For calculating AUROC during training
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (features, masks, delta_t, targets) in enumerate(pbar):
            features = features.to(self.device)
            masks = masks.to(self.device)
            delta_t = delta_t.to(self.device)
            targets = targets.to(self.device)
            
            use_cuda_amp = (self.device == 'cuda')
            with torch.autocast(device_type=('cuda' if use_cuda_amp else 'cpu'), dtype=torch.float16, enabled=use_cuda_amp):
                # Forward pass
                outputs = self.model(features, masks, delta_t)
                
                # Squeeze outputs if necessary
                if outputs.dim() > 1 and targets.dim() == 1:
                    outputs = outputs.squeeze(-1)

                # Calculate loss
                loss = criterion(outputs, targets.float())
            
            # Backward pass
            optimizer.zero_grad()
            if use_cuda_amp:
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_value)
                optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            
            # Store predictions for metrics calculation
            with torch.no_grad():
                preds = torch.sigmoid(outputs).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets_cpu.flatten())
            
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / total_samples
        # Ensure float32 for MPS compatibility
        metrics = self._calculate_metrics(
            np.array(all_targets, dtype=np.float32), 
            np.array(all_preds, dtype=np.float32)
        )
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader, criterion) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for features, masks, delta_t, targets in pbar:
                features = features.to(self.device)
                masks = masks.to(self.device)
                delta_t = delta_t.to(self.device)
                targets = targets.to(self.device)
                
                use_cuda_amp = (self.device == 'cuda')
                with torch.autocast(device_type=('cuda' if use_cuda_amp else 'cpu'), dtype=torch.float16, enabled=use_cuda_amp):
                    outputs = self.model(features, masks, delta_t)

                    if outputs.dim() > 1 and targets.dim() == 1:
                        outputs = outputs.squeeze(-1)
                        
                    loss = criterion(outputs, targets.float())
                
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
                preds = torch.sigmoid(outputs).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                all_preds.extend(preds.flatten())
                all_targets.extend(targets_cpu.flatten())
        
        avg_loss = total_loss / total_samples
        # Ensure float32 for MPS compatibility
        metrics = self._calculate_metrics(
            np.array(all_targets, dtype=np.float32),
            np.array(all_preds, dtype=np.float32)
        )
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, targets: np.ndarray, preds: np.ndarray) -> Dict:
        """Calculate all metrics"""
        return calculate_all_metrics(targets, preds)
    
    def train(self, train_loader, val_loader, 
              epochs: int = 60, 
              optimizer_config: Dict = {'name': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4},
              scheduler_config: Dict = {'name': 'CosineAnnealing', 'warmup_epochs': 3, 'T_max': 57},
              loss_config: Dict = {'name': 'BCE', 'pos_weight': 1.0, 'label_smoothing': 0.0,
                                   'focal_alpha': 0.25, 'focal_gamma': 2.0},
              early_stopping_config: Dict = {'patience': 8, 'metric': 'auprc'},
              grad_clip_value: float = 1.0,
              save_path: Optional[str] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
            loss_config: Loss function configuration
            early_stopping_config: Early stopping configuration
            grad_clip_value: Gradient clipping value
            save_path: Path to save model weights
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        if optimizer_config['name'] == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=optimizer_config['lr'],
                                    weight_decay=optimizer_config['weight_decay'])
        else: # Default to Adam
            optimizer = optim.Adam(self.model.parameters(), lr=optimizer_config['lr'],
                                   weight_decay=optimizer_config.get('weight_decay', 1e-5))

        # Setup scheduler
        if scheduler_config['name'] == 'CosineAnnealing':
            T_max = epochs - scheduler_config['warmup_epochs']
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)
        else: # Default to ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Warmup scheduler
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        # Loss function
        pos_weight_tensor = torch.tensor([float(loss_config['pos_weight'])], dtype=torch.float32, device=self.device)
        if loss_config['name'] == 'Focal':
            criterion = FocalLoss(alpha=loss_config['focal_alpha'], gamma=loss_config['focal_gamma'],
                                  pos_weight=pos_weight_tensor)
        elif loss_config.get('label_smoothing', 0.0) > 0:
            criterion = LabelSmoothingLoss(smoothing=loss_config['label_smoothing'],
                                           pos_weight=pos_weight_tensor)
        else: # BCE
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_config['patience'])
        es_metric = early_stopping_config['metric']
        
        # Grad scaler for mixed precision (CUDA only)
        use_cuda_amp = (self.device == 'cuda')
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
        
        best_val_score = 0.0
        
        for epoch in range(epochs):
            # Warmup phase
            if epoch < warmup_epochs:
                lr_scale = (epoch + 1) / max(1, warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = optimizer_config['lr'] * lr_scale
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, criterion, optimizer, scaler, grad_clip_value)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            if epoch >= warmup_epochs:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics[es_metric])
                else:
                    scheduler.step()

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_scores.append(train_metrics)
            self.val_scores.append(val_metrics)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train AUPRC: {train_metrics["auprc"]:.4f}, Train AUROC: {train_metrics["auroc"]:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val AUPRC: {val_metrics["auprc"]:.4f}, Val AUROC: {val_metrics["auroc"]:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if early_stopping(val_metrics[es_metric], self.model):
                print(f'Early stopping at epoch {epoch+1} based on {es_metric}')
                break
            
            # Save best model
            if val_metrics[es_metric] > best_val_score:
                best_val_score = val_metrics[es_metric]
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
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon GPU)")
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
