from transformers import PretrainedConfig

class TransformerConfig(PretrainedConfig):

    def __init___(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            hidden_size: int = 512,
            max_position_embeddings: int = 512,
            num_layers: int = 6,
            num_attention_heads: int = 8,
            attention_dropout: float = 0.1,
            intermediate_size: int = 2048,
            hidden_dropout: float = 0.1,
            qkv_bias: bool = False,
            **kwargs
    ) -> None:

        """
        Args:
            src_vocab_size (int): Source vocabulary size.
            tgt_vocab_size (int): Target vocabulary size.
            hidden_size (int, optional): Hidden size of the model. Defaults to 512.
            max_position_embeddings (int, optional): Maximum position embeddings. Defaults to 512.
            num_layers (int, optional): Number of layers in the transformer. Defaults to 6.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 8.
            intermediate_size (int, optional): Size of the intermediate layer. Defaults to 2048.
            hidden_dropout (float, optional): Dropout rate for hidden layers. Defaults to 0.1.
            qkw_bias (bool, optional): Whether to use bias in query and key projections. Defaults to False.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.qkv_bias = qkv_bias