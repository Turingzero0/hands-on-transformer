import torch
from torch import nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Size of the hidden layer.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.scaling = hidden_size ** 0.5 # sqrt(d_k)

    def forward(self, token: Tensor) -> Tensor:
        """
        Args:
            token (Tensor): Input token indices of shape (batch_size, seq_len).

        Returns:
            Tensor: Embedded tokens of shape (batch_size, seq_len, hidden_size).
        """
        embedded = self.embedding(token) * self.scaling
        return embedded