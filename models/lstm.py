"""
Bidirectional LSTM Model for sequential ICU data
"""
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """
    Bidirectional LSTM Model for sequential modeling of ICU time-series data
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM - not using bidirectional for this task to avoid data leakage from future
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, masks: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: [batch_size, seq_len, input_size] - input features
            masks: [batch_size, seq_len, input_size] - mask indicating observed values
            delta_t: [batch_size, seq_len, 1] - time since last observation (not used in this model)
            
        Returns:
            Logits: [batch_size, 1]
        """
        if delta_t is None:  # regular hourly sampling: 0 for the first step, 1 afterwards
            delta_t = torch.ones(features.shape[0], features.shape[1], 1, device=features.device)
            delta_t[:, 0] = 0
        # Apply mask to features (set missing values to 0) - imputation should be handled in data loader
        masked_features = features
        
        # LSTM processing
        lstm_out, _ = self.lstm(masked_features)
        
        # Use last hidden state (concatenated forward and backward)
        last_hidden = lstm_out[:, -1, :]
        
        # Output layer
        output = self.fc(self.dropout(last_hidden))
        
        return output
