from transformers import WhisperConfig

from apex_plus.ir.block import Block
from apex_plus.ir.cells.attention import MHA, BiMHA
from apex_plus.ir.cells.ffn import MLP
from apex_plus.ir.transformer import Transformer
from apex_plus.models.model import ApexModel


class Whisper(ApexModel):

    def __init__(
        self,
        vocab_size: int,
        num_encoder_layers: int,
        num_encoder_heads: int,
        encoder_hidden_size: int,
        encoder_intermediate_size: int,
        num_decoder_layers: int,
        num_decoder_heads: int,
        decoder_hidden_size: int,
        decoder_intermediate_size: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_intermediate_size = encoder_intermediate_size
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_intermediate_size = decoder_intermediate_size

        if encoder_hidden_size % num_encoder_heads != 0:
            raise ValueError(
                f"encoder_hidden_size {encoder_hidden_size} must be divisible by "
                f"num_encoder_heads {num_encoder_heads}"
            )
        self.head_size = encoder_hidden_size // num_encoder_heads

        if decoder_hidden_size % num_decoder_heads != 0:
            raise ValueError(
                f"decoder_hidden_size {decoder_hidden_size} must be divisible by "
                f"num_decoder_heads {num_decoder_heads}"
            )
        self.head_size = decoder_hidden_size // num_decoder_heads

    @classmethod
    def from_hf(cls, config: WhisperConfig) -> "Whisper":
        return cls(
            vocab_size=config.vocab_size,
            num_encoder_layers=config.encoder_layers,
            num_encoder_heads=config.encoder_attention_heads,
            encoder_hidden_size=config.encoder_ffn_dim,
            encoder_intermediate_size=config.encoder_ffn_dim,
            num_decoder_layers=config.decoder_layers,
            num_decoder_heads=config.decoder_attention_heads,
            decoder_hidden_size=config.decoder_ffn_dim,
            decoder_intermediate_size=config.decoder_ffn_dim,
        )

    def to_ir(self) -> Transformer:

        enc_bimha = BiMHA(self.num_encoder_heads, self.encoder_hidden_size)
        enc_mlp = MLP(self.encoder_hidden_size, self.encoder_intermediate_size)
        encoder_block = Block(cells=[enc_bimha, enc_mlp])

        dec_mha = MHA(self.num_decoder_heads, self.decoder_hidden_size)
        dec_bimha = BiMHA(self.num_decoder_heads, self.decoder_hidden_size)
        dec_mlp = MLP(self.decoder_hidden_size, self.decoder_intermediate_size)
        decoder_block = Block(cells=[dec_mha, dec_bimha, dec_mlp])

        return Transformer.from_blocks(
            vocab_size=self.vocab_size,
            hidden_size=self.encoder_hidden_size,
            num_encoder_blocks=self.num_encoder_layers,
            encoder_block=encoder_block,
            num_decoder_blocks=self.num_decoder_layers,
            decoder_block=decoder_block,
        )

    def __repr__(self) -> str:
        return (
            f"Whisper(vocab_size={self.vocab_size}, "
            f"num_encoder_layers={self.num_encoder_layers}, "
            f"num_encoder_heads={self.num_encoder_heads}, "
            f"num_decoder_layers={self.num_decoder_layers}, "
            f"num_decoder_heads={self.num_decoder_heads}, "
            f"hidden_size={self.encoder_hidden_size}, "
            f"intermediate_size={self.encoder_hidden_size})"
        )
