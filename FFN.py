import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import transformer_config
from transformer_config import TransformerConfig

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the FeedForwardNetwork with the given configuration."""

        super().__init__()

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        """Initialize the LayerNorm with the given normalized shape and epsilon."""
        super().__init__()
        # self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps) 
        # # Using nn.LayerNorm directly is more efficient and recommended
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        return self.gamma * normalized_x + self.beta


class TransformerOutputLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:

        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)