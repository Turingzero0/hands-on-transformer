import torch
from torch import nn
from torch import Tensor
import transformer_config
from transformer_config import TransformerConfig

class PositionEncoding(nn.Module):
    def __init__(self,config: TransformerConfig) -> None:
        """
        Args:
            config (TransformerConfig): Configuration object containing model parameters.
        """
        super().__init__()
        
        # shape: [max_len,1]
        pos = torch.arange(0, config.max_position_embeddings, dtype=torch.float).unsqueeze(1)# [0,1,2,...,max_position_embeddings-1] -> [max_position_embeddings,1]
        div_term = 10000 ** (-torch.arange(0, config.hidden_size , step = 2, dtype=torch.float) / config.hidden_size) # [1,2,3,...,hidden_size-1] -> [hidden_size/2]
        pe = torch.zeros(config.max_position_embeddings, config.hidden_size, dtype=torch.float) # [max_position_embeddings,hidden_size]
        pe[:, 0::2] = torch.sin(pos * div_term) # even indices
        pe[:, 1::2] = torch.cos(pos * div_term) # odd indices
        pe = pe.unsqueeze(0) # [1,max_position_embeddings,hidden_size]
        self.register_buffer('pe', pe) # register as buffer to avoid being treated as a parameter

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:,:x.size()[1]]
