import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformer_config import TransformerConfig
from position_encoding import PositionEncoding
from Multi_Head_Attention import MultiHeadAttention
from FFN import FeedForwardNetwork, LayerNorm, TransformerOutputLayer
from Encoder import TransformerEncoderLayer
from Encoder import TransformerEncoder
from Decoder import TransformerDecoderLayer
from Decoder import TransformerDecoder
from typing import Optional

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.max_length = config.max_position_embeddings
        self.input_embedding = nn.Embedding(config.src_vocab_size, config.hidden_size)
        self.output_embedding = nn.Embedding(config.tgt_vocab_size, config.hidden_size)
        self.position_encoding = PositionEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.output = TransformerOutputLayer(config)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in  self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def encode(
            self,
            src: Tensor,
            src_key_padding_mask: Optional[Tensor] = None,
            src_attn_mask: Optional[Tensor] = None      
    ) -> tuple:
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, src_seq_len).
            src_key_padding_mask (Optional[Tensor]): Mask for padding tokens.
            src_attn_mask (Optional[Tensor]): Attention mask.
        
        Returns:
            tuple: Encoder output and attention weights.
        """
        src = self.input_embedding(src)
        src = self.position_encoding(src)

        encoder_output = self.encoder(
            src=src,
            sec_key_padding_mask=src_key_padding_mask,
            src_attn_mask=src_attn_mask
        )
        return encoder_output
    
    def decode(
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
            tgt (Tensor): Target tensor of shape (batch_size, tgt_seq_len).
            memory (Tensor): Encoder output of shape (batch_size, src_seq_len, hidden_size).
            tgt_key_padding_mask (Optional[Tensor]): Mask for target padding tokens.
            tgt_attn_mask (Optional[Tensor]): Mask for target self-attention.
            memory_key_padding_mask (Optional[Tensor]): Mask for encoder output padding tokens.
            memory_attn_mask (Optional[Tensor]): Mask for encoder self-attention.
        
        Returns:
            Tensor: Decoder output of shape (batch_size, tgt_seq_len, hidden_size).
        """
        tgt = self.output_embedding(tgt)
        tgt = self.position_encoding(tgt)

        decoder_output = self.decoder(
            tgt=tgt,
            memory=memory_tuple,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_attn_mask=tgt_attn_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_attn_mask=memory_attn_mask
        )
        return decoder_output
    
    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_key_padding_mask: Optional[Tensor] = None,
            src_attn_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            tgt_attn_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            memory_attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, src_seq_len).
            tgt (Tensor): Target tensor of shape (batch_size, tgt_seq_len).
            src_key_padding_mask (Optional[Tensor]): Mask for padding tokens in source.
            src_attn_mask (Optional[Tensor]): Attention mask for source.
            tgt_key_padding_mask (Optional[Tensor]): Mask for padding tokens in target.
            tgt_attn_mask (Optional[Tensor]): Attention mask for target.
            memory_key_padding_mask (Optional[Tensor]): Mask for encoder output padding tokens.
            memory_attn_mask (Optional[Tensor]): Mask for encoder self-attention.

        Returns:
            Tensor: Output tensor of shape (batch_size, tgt_seq_len, hidden_size).
        """
        encoder_output = self.encode(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            src_attn_mask=src_attn_mask
        )
        
        decoder_output = self.decode(
            tgt=tgt,
            memory_tuple=encoder_output,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_attn_mask=tgt_attn_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_attn_mask=memory_attn_mask
        )
        
        output = self.output(decoder_output)
        return output

    @torch.inference_mode()
    def inference(
        self,
        src: Tensor,
        tgt_start_token_id: int,
        tgt_end_token_id: int,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> list:
        assert src.shape[0] == 1,"batch_size must be 1 for inference"
        device = src.device

        memory_tuple = self.encode(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            src_attn_mask=src_attn_mask
        )
        tgt_tokens = torch.tensor([[tgt_start_token_id]], device=device)

        for _ in range(self.max_length):
            tgt = torch.LongTensor([tgt_tokens]).to(device)
            tgt_padding_mask = torch.zeros_like(tgt, dtype=torch.bool, device=device)
            causal_mask = torch.tril(
                torch.ones((1, tgt.shape[1], tgt.shape[1]), device=device)
            ).bool()
            decoder_output = self.decode(
                tgt=tgt,
                memory_tuple=memory_tuple,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_attn_mask=causal_mask,
                memory_key_padding_mask=None,
                memory_attn_mask=None
            )[0]
            next_token_logits = self.output(decoder_output)
            next_token = next_token_logits.argmax(dim=-1)[-1].item()

            if next_token == tgt_end_token_id:
                break
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]], device=device)], dim=1)

        return tgt_tokens.squeeze(0).tolist()  # Convert to list and remove batch dimension