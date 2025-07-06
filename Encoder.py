import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformer_config import TransformerConfig
from Multi_Head_Attention import MultiHeadAttention
from FFN import FeedForwardNetwork, LayerNorm
from typing import Optional

class TransformerEncoderLayer(nn.Module):
    def __init__(self,config: TransformerConfig) -> None:
        super.__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size)
        self.ffn = FeedForwardNetwork(config)
        self.norm2 = LayerNorm(config.hidden_size)
    
    def forward(self,
                src: Tensor,
                sec_key_padding_mask: Optional[Tensor] = None,
                src_attn_mask: Optional[Tensor] = None
                ) -> Tensor:
        
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            sec_key_padding_mask (Optional[Tensor], optional): Mask for padding tokens. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): Attention mask. Defaults to None.

        Returns:   
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        residual = src
        x = self.attention(
            q=src,
            k=src,
            v=src,
            key_padding_mask=sec_key_padding_mask,
            attn_mask=src_attn_mask
        )
        x = self.norm1(x + residual)  # Add & Norm

        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)  # Add & Norm
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self,
                src: Tensor,
                sec_key_padding_mask: Optional[Tensor] = None,
                src_attn_mask: Optional[Tensor] = None
                ) -> tuple:
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            sec_key_padding_mask (Optional[Tensor], optional): Mask for padding tokens. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): Attention mask. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        x = src
        all_hidden_states = ()
        for layer in self.layers:
            x = layer(x, sec_key_padding_mask, src_attn_mask)
            
            all_hidden_states += (x,)

        return all_hidden_states