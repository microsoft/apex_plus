from transformers import CLIPConfig

from apex_plus.ir.block import Block
from apex_plus.ir.cells.attention import BiMHA
from apex_plus.ir.cells.ffn import MLP
from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel


class CLIPVision(ApexModel):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        image_size: int,
        patch_size: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"num_heads {num_heads}"
            )

        self.head_size = hidden_size // num_heads

    @classmethod
    def from_hf(cls, config: CLIPConfig) -> "CLIPVision":
        v_config = config.vision_config
        # there is also text encoder in CLIP, so we extract the vision encoder
        return cls(
            hidden_size=v_config.hidden_size,
            intermediate_size=v_config.intermediate_size,
            num_layers=v_config.num_hidden_layers,
            num_heads=v_config.num_attention_heads,
            num_channels=v_config.num_channels,
            image_size=v_config.image_size,
            patch_size=v_config.patch_size,
        )

    def to_ir(self) -> Transformer:
        mha = BiMHA(self.num_heads, self.hidden_size)
        mlp = MLP(self.hidden_size, self.intermediate_size)

        encoder_block = Block(cells=[mha, mlp])
        return Transformer.from_blocks(
            vocab_size=0,  # TODO check what should be the apporpriate val
            hidden_size=self.hidden_size,
            num_encoder_blocks=self.num_layers,
            encoder_block=encoder_block,
            num_decoder_blocks=0,
            decoder_block=None,
        )

    def __repr__(self) -> str:
        return (
            f"CLIPVision(num_channels={self.num_channels}, "
            f"image_size={self.image_size}, "
            f"patch_size={self.patch_size}, "
            f"num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
