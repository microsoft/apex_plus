from transformers import LlamaConfig

from apex_plus.ir.block import Block
from apex_plus.ir.cells.attention import MQA
from apex_plus.ir.cells.ffn import SwiGLU
from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel


class LLaMA3(ApexModel):

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"num_heads {num_heads}"
            )
        self.head_size = hidden_size // num_heads

    @classmethod
    def from_hf(cls, config: LlamaConfig) -> "LLaMA3":
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def to_ir(self) -> Transformer:
        print(self.__repr__)

        mqa = MQA(self.num_heads, self.num_kv_heads, self.head_size, self.hidden_size)
        mlp = SwiGLU(self.hidden_size, self.intermediate_size)
        decoder_block = Block(cells=[mqa, mlp])
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
            f"LLaMA3(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size})"
        )
