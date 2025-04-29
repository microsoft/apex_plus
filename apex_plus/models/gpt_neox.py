from transformers import GPTNeoXConfig

from apex_plus.ir.block import Block
from apex_plus.ir.cells.attention import MHA, ParallelMHAMLP
from apex_plus.ir.cells.ffn import MLP
from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel


class GPTNeoX(ApexModel):

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        intermediate_size: int,
        parallel_attn: bool,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.parallel_attn = parallel_attn

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"num_heads {num_heads}"
            )
        self.head_size = hidden_size // num_heads

    @classmethod
    def from_hf(cls, config: GPTNeoXConfig) -> "GPTNeoX":
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            parallel_attn=config.use_parallel_residual,
        )

    def to_ir(self) -> Transformer:
        if self.parallel_attn:
            layer = ParallelMHAMLP(
                self.num_heads, self.hidden_size, self.intermediate_size
            )
            decoder_block = Block(cells=[layer])
        else:
            mha = MHA(self.num_heads, self.hidden_size)
            mlp = MLP(self.hidden_size, self.intermediate_size)
            decoder_block = Block(cells=[mha, mlp])
        return Transformer.from_blocks(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_blocks=0,
            encoder_block=None,
            num_decoder_blocks=self.num_layers,
            decoder_block=decoder_block,
        )

    def __repr__(self) -> str:
        return (
            f"GPTNeoX(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"parallel_attn={self.parallel_attn})"
        )
