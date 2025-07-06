import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformer_config import TransformerConfig
from Multi_Head_Attention import MultiHeadAttention
from FFN import FeedForwardNetwork, LayerNorm
from typing import Optional

class TransformerDecoderLayer(nn.Module):
    def _init__(self, config:TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size)

        self.cross_attention = MultiHeadAttention(config)
        self.norm2 = LayerNorm(config.hidden_size)

        self.ffn = FeedForwardNetwork(config)
        self.norm3 = LayerNorm(config.hidden_size)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_attn_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        
        """
            Args:
                tgt (Tensor): Target sequence of shape (batch_size, tgt_seq_len, hidden_size).
                memory (Tensor): Encoder output of shape (batch_size, src_seq_len, hidden_size).
                tgt_key_padding_mask (Optional[Tensor]): Mask for target padding tokens.
                tgt_attn_mask (Optional[Tensor]): Mask for target self-attention.
                memory_key_padding_mask (Optional[Tensor]): Mask for encoder output padding tokens.
                memory_attn_mask (Optional[Tensor]): Mask for encoder self-attention.

            Returns:
                Tensor: Output of the decoder layer of shape (batch_size, tgt_seq_len, hidden_size).
        """

        residual = tgt
        x = self.self_attention(
            q = tgt,
            k = tgt,
            v = tgt,
            key_padding_mask = tgt_key_padding_mask,
            attn_mask = tgt_attn_mask
        )
        x = self.norm1(x + residual)

        residual = x
        x = self.cross_attention(
            q = x,
            k = memory,
            v = memory,
            key_padding_mask = memory_key_padding_mask,
            attn_mask = memory_attn_mask
        )
        x = self.norm2(x + residual)

        residual = x
        x = self.ffn(x)
        x = self.norm3(x + residual)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self,config: TransformerConfig)-> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
            self,
            tgt: Tensor,
            memory_tuple: tuple,
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_attn_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        
        """
            Args:
                tgt (Tensor): Target sequence of shape (batch_size, tgt_seq_len, hidden_size).
                memory_tuple (tuple): Tuple containing encoder output and its mask.
                tgt_key_padding_mask (Optional[Tensor]): Mask for target padding tokens.
                tgt_attn_mask (Optional[Tensor]): Mask for target self-attention.
                memory_key_padding_mask (Optional[Tensor]): Mask for encoder output padding tokens.
                memory_attn_mask (Optional[Tensor]): Mask for encoder self-attention.
            Returns:
                Tensor: Output of the decoder of shape (batch_size, tgt_seq_len, hidden_size).
        """

        x = tgt
        for i, layer in enumerate(self.layers):
            x = layer(
                tgt=x,
                memory=memory_tuple[i],
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_attn_mask=tgt_attn_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_attn_mask=memory_attn_mask
            )

        return x
