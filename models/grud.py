"""
GRU-D Model: Handles missing data with time-decay mechanism
"""
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUD(nn.Module):
    """
    GRU-D Model: Gated Recurrent Unit with Decay mechanism for handling missing data
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Decay parameters for mean and features
        self.decay_gamma_x = nn.Parameter(torch.rand(input_size))
        self.decay_gamma_h = nn.Parameter(torch.rand(hidden_size))
        
        # GRU Cell
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size) # Input is concat of decayed_x and mask
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, seq_len, input_size] - input features (imputed)
            mask: [batch_size, seq_len, input_size] - mask indicating observed values
            delta_t: [batch_size, seq_len, 1] - time since last observation for each feature
            
        Returns:
            Logits: [batch_size, 1]
        """
        if delta_t is None:  # regular hourly sampling: 0 for the first step, 1 afterwards
            delta_t = torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
            delta_t[:, 0] = 0
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Store mean of features for imputation (can be precomputed)
        x_mean = torch.mean(x, dim=[0, 1]) 
        
        for t in range(seq_len):
            x_t = x[:, t, :]       # current input [B, F]
            m_t = mask[:, t, :].float()  # current mask as float [B, F]
            d_t = delta_t[:, t, :]   # current delta_t [B, 1]
            
            # Feature-level decay
            gamma_x = torch.exp(-torch.relu(self.decay_gamma_x))              # [F]
            d_t_feat = d_t.expand(-1, x_t.shape[1])                           # [B, F]
            decayed_gamma_x = torch.pow(gamma_x, d_t_feat)                    # [B, F]
            
            # Impute using decayed mean
            x_imputed = m_t * x_t + (1.0 - m_t) * (decayed_gamma_x * x_mean)  # [B, F]
            
            # Hidden state decay
            gamma_h = torch.exp(-torch.relu(self.decay_gamma_h))              # [H]
            d_t_h = d_t.mean(dim=1, keepdim=True)                             # [B, 1]
            decayed_gamma_h = torch.pow(gamma_h, d_t_h)                       # [B, H]
            h = h * decayed_gamma_h

            # Concatenate imputed features and mask
            x_combined = torch.cat([x_imputed, m_t], dim=1)                   # [B, 2F]
            
            # Update hidden state
            h = self.gru_cell(x_combined, h)
            
        # Output layer
        output = self.fc(self.dropout(h))
        
        return output
