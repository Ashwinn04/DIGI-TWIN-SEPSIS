"""
CNN-LSTM Hybrid Model: Combines local pattern extraction with temporal modeling
"""
import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid Model: Uses CNN for local pattern extraction, then LSTM for temporal modeling
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.3, cnn_filters: List[int] = [64, 128], kernel_size: int = 3):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            cnn_filters: List of output channels for CNN blocks
            kernel_size: CNN kernel size
        """
        super(CNNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 1D CNN blocks for local pattern extraction
        cnn_blocks = []
        in_channels = input_size
        for out_channels in cnn_filters:
            cnn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )
            in_channels = out_channels
        self.cnn_blocks = nn.Sequential(*cnn_blocks)
        
        # Calculate LSTM input size after CNN blocks
        # Each maxpool layer with stride 2 halves the sequence length
        final_cnn_channels = cnn_filters[-1]
        self.lstm = nn.LSTM(final_cnn_channels, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: torch.Tensor, masks: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: [batch_size, seq_len, input_size] - input features
            masks: [batch_size, seq_len, input_size] - mask indicating observed values
            delta_t: Time delta (not used in this model)
            
        Returns:
            Logits: [batch_size, 1]
        """
        if delta_t is None:  # regular hourly sampling: 0 for the first step, 1 afterwards
            delta_t = torch.ones(features.shape[0], features.shape[1], 1, device=features.device)
            delta_t[:, 0] = 0
        # Reshape for CNN: [batch, seq_len, features] -> [batch, features, seq_len]
        cnn_input = features.transpose(1, 2)
        
        # CNN processing
        cnn_out = self.cnn_blocks(cnn_input)
        
        # Reshape back for LSTM: [batch, filters, seq_len'] -> [batch, seq_len', filters]
        lstm_input = cnn_out.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Output layer
        output = self.fc(self.dropout(last_hidden))
        
        return output
