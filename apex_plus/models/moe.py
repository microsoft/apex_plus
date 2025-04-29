from transformers import OPTConfig

from apex_plus.ir.block import Block
from apex_plus.ir.cells.attention import MHA
from apex_plus.ir.cells.ffn import MLP, MoE, SwiMoE
from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel


class OPTMoE(ApexModel):

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        capacity_factor: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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
        self.head_size = hidden_size // num_heads

    @classmethod
    def from_hf(
        cls,
        config: OPTConfig,
        num_experts: int,
        topk: int,
        capacity_factor: float,
    ) -> "MoE":
        return cls(
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            intermediate_size=4 * config.hidden_size,
            num_experts=num_experts,
            topk=topk,
            capacity_factor=capacity_factor,
        )

    def to_ir(self) -> Transformer:
        assert self.num_layers % 2 == 0
        mha0 = MHA(self.num_heads, self.hidden_size)
        mlp = MLP(self.hidden_size, self.intermediate_size)
        mha1 = MHA(self.num_heads, self.hidden_size)
        moe = MoE(
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            self.topk,
            self.capacity_factor,
        )
        decoder_block = Block(cells=[mha0, mlp, mha1, moe])
        return Transformer.from_blocks(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_blocks=0,
            encoder_block=None,
            num_decoder_blocks=self.num_layers // 2,
            decoder_block=decoder_block,
        )

    def __repr__(self) -> str:
        return (
            f"OPTMoE(vocab_size={self.vocab_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"num_experts={self.num_experts}, "
            f"topk={self.topk}, "
            f"capacity_factor={self.capacity_factor})"
        )
