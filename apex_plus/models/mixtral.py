from transformers import MixtralConfig

from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel
from apex_plus.ir.cells.attention import MQA
from apex_plus.ir.cells.ffn import MLP
from apex_plus.ir.cells.ffn import SwiMoE
from apex_plus.ir.block import Block


class Mixtral(ApexModel):

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int,
        topk: int,
        capacity_factor: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.topk = topk
        self.capacity_factor = capacity_factor

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"num_heads {num_heads}"
            )
        if num_experts <= topk:
            raise ValueError(
                f"num_experts {num_experts} must be larger than " f"topk {topk}."
            )
        self.head_size_mqa = hidden_size // num_heads

    @classmethod
    def from_hf(
        cls,
        config: MixtralConfig,
        num_experts: int,
        topk: int,
        capacity_factor: float,
    ) -> "Mixtral":
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            # 8 experts for Mixtral-8x7B
            num_experts=config.num_local_experts,
            # K has a value of 2 for Mixtral-8x7B
            topk=config.num_experts_per_tok,
            capacity_factor=capacity_factor,
        )

    def to_ir(self) -> Transformer:
        # NOTE: MQA is currently used for Mixtral but can workt with MHA or GQA
        mqa = MQA(
            self.num_heads, self.num_kv_heads, self.head_size_mqa, self.hidden_size
        )
        mlp = MLP(self.hidden_size, self.intermediate_size)
        swimoe = SwiMoE(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            self.topk,
            self.capacity_factor,
        )
        decoder_block = Block(cells=[mqa, mlp, swimoe])

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
            f"Mixtral(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"num_experts={self.num_experts}, "
            f"topk={self.topk}, "
            f"capacity_factor={self.capacity_factor})"
        )
