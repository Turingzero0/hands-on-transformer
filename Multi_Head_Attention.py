import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import transformer_config
from transformer_config import TransformerConfig

def attention_forward(
        module: nn.Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scaling: float,
        dropout: float = 0.0,
        mask: Tensor = None
) -> Tensor:
    """
    Args:
        module (nn.Module): The module containing the attention mechanism.
        query (Tensor): Query tensor of shape (batch_size, seq_len, hidden_size).
        key (Tensor): Key tensor of shape (batch_size, seq_len, hidden_size).
        value (Tensor): Value tensor of shape (batch_size, seq_len, hidden_size).
        scaling (float): Scaling factor for the attention scores.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        mask (Tensor, optional): Mask tensor for attention. Defaults to None.

    Returns:
        Tensor: Output tensor after applying attention.
    """
    
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling  # [batch_size, seq_len, seq_len]
    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(attn_weights, dim=-1)  # [batch_size, seq_len, seq_len]
    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    output = torch.matmul(attn_weights, value)  # [batch_size, seq_len, hidden_size]
    return output


def transpose_for_scores(self, x:Tensor) -> Tensor:
    """
    Transpose the input tensor for multi-head attention.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

    Returns:
        Tensor: Transposed tensor of shape (batch_size, num_heads, seq_len, head_dim).
    """
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # add two new dimensions, now (batch_size, seq_len, num_attention_heads, attention_head_size)
    # hidden_size = num_attention_heads * head_dim
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]

class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:

        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size ** (-0.5)
        self.attention_dropout = config.attention_dropout

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.linear = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            key_padding_mask: Tensor = None,
            attn_mask: Tensor = None
    ) -> Tensor:
        
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, hidden_size).
            k (Tensor): Key tensor of shape (batch_size, seq_len, hidden_size).
            v (Tensor): Value tensor of shape (batch_size, seq_len, hidden_size).
            key_padding_mask (Tensor, optional): Mask for padding tokens. Defaults to None.
            attn_mask (Tensor, optional): Attention mask. Defaults to None.
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        
        q = self.transpose_for_scores(self.query(q))  # [batch_size, num_heads, seq_len, head_dim]
        # Q= W_q * X, where W_q is the weight matrix for query and X is the input tensor
        k = self.transpose_for_scores(self.key(k))    # [batch_size, num_heads, seq_len, head_dim]
        # K= W_k * X, where W_k is the weight matrix for key and X is the input tensor
        v = self.transpose_for_scores(self.value(v)) # [batch_size, num_heads, seq_len, head_dim]
        # V= W_v * X, where W_v is the weight matrix for value and X is the input tensor

        mask = self.merge_masks(key_padding_mask, attn_mask) if key_padding_mask is not None or attn_mask is not None else None 
        # merge_masks combines key_padding_mask and attn_mask into a single mask

        output = attention_forward(
            self,
            query=q,
            key=k,
            value=v,
            scaling=self.scaling,
            dropout= 0.0 if self.training else self.attention_dropout,
            mask=mask
        )

        output = output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        new_size = output.size()[:2] + (self.all_head_size,)  # [batch_size, seq_len, all_head_size]
        output = output.reshape(new_size)  # reshape to [batch_size, seq_len, hidden_size]
        output = self.linear(output)
        output = self.dropout(output)

        return output


def merge_masks(
        self, key_padding_mask: Tensor, attn_mask: Tensor
):
    """
    Merges key_padding_mask and attn_mask into a single mask.

    Args:
        key_padding_mask (Tensor): Mask for padding tokens, size (batch_size, seq_len).
        attn_mask (Tensor): Attention mask, size (seq_len, seq_len), because padding has been applied to the input sequence.

    Returns:
        Tensor: Merged mask, size (batch_size, num_heads, seq_len, seq_len).
    """
    if key_padding_mask is None and attn_mask is None:
        return None
    
    mask = None

    if key_padding_mask is not None:
        # shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len] -> [batch_size, num_heads, 1, seq_len]
        bz, len_k = key_padding_mask.shape
        key_padding_mask = key_padding_mask.view(bz,1,1,len_k).expand(-1, self.num_attention_heads, -1, -1)  # [batch_size, num_heads, 1, seq_len]
        mask = key_padding_mask

    if attn_mask is not None:
        # shape: [seq_len, seq_len] -> [1, 1, seq_len, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        attn_mask = attn_mask.view(1, 1, attn_mask.size(0), attn_mask.size(1)).expand(bz, self.num_attention_heads, -1, -1)

        if mask is None:
            mask = attn_mask
        else:
            mask = mask.logical_or(attn_mask)  # Combine the two masks

    return mask

    
    