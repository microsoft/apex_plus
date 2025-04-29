from typing import Optional

from apex_plus.ir.block import Block
from apex_plus.ir.cells.embedding import Embedding
from apex_plus.ir.cells.sampler import Sampler


class Transformer:

    def __init__(
        self,
        embedding: Embedding,
        num_encoder_blocks: int,
        encoder_block: Optional[Block],
        num_decoder_blocks: int,
        decoder_block: Block,
        sampler: Sampler,
        hidden_size: int,
    ) -> None:
        self.embedding = embedding
        self.num_encoder_blocks = num_encoder_blocks
        self.encoder_block = encoder_block
        self.num_decoder_blocks = num_decoder_blocks
        self.decoder_block = decoder_block
        self.sampler = sampler
        self.hidden_size = hidden_size

        if self.num_encoder_blocks < 0:
            raise ValueError(
                f"num_encoder_blocks must be non-negative, "
                f"got {self.num_encoder_blocks}"
            )
        elif self.num_encoder_blocks == 0:
            if self.encoder_block is not None:
                raise ValueError(
                    f"encoder_block must be None if "
                    "num_encoder_blocks == 0, got "
                    f"{self.encoder_block}"
                )
        else:
            # num_encoder_blocks > 0
            if self.encoder_block is None:
                raise ValueError(
                    f"encoder_block must not be None if "
                    "num_encoder_blocks > 0, got "
                    f"{self.encoder_block}"
                )

        if self.num_decoder_blocks < 0:
            raise ValueError(
                f"num_decoder_blocks must be non-negative, "
                f"got {self.num_decoder_blocks}"
            )
        elif self.num_decoder_blocks == 0:
            if self.decoder_block is not None:
                raise ValueError(
                    f"decoder_block must be None if "
                    "num_decoder_blocks == 0, got "
                    f"{self.decoder_block}"
                )
        else:
            # num_decoder_blocks > 0
            if self.decoder_block is None:
                raise ValueError(
                    f"decoder_block must not be None if "
                    "num_decoder_blocks > 0, got "
                    f"{self.decoder_block}"
                )

    @classmethod
    def from_blocks(
        cls,
        vocab_size: int,
        hidden_size: int,
        num_encoder_blocks: int,
        encoder_block: Optional[Block],
        num_decoder_blocks: int,
        decoder_block: Block,
    ) -> "Transformer":
        embedding = Embedding(vocab_size, hidden_size)
        sampler = Sampler(vocab_size, hidden_size)
        return cls(
            embedding,
            num_encoder_blocks,
            encoder_block,
            num_decoder_blocks,
            decoder_block,
            sampler,
            hidden_size,
        )

    def __repr__(self) -> str:
        return (
            f"Transformer(embedding={self.embedding}, "
            f"num_encoder_blocks={self.num_encoder_blocks}, "
            f"encoder_block={self.encoder_block}, "
            f"num_decoder_blocks={self.num_decoder_blocks}, "
            f"decoder_block={self.decoder_block}, "
            f"sampler={self.sampler}, "
            f"hidden_size={self.hidden_size})"
        )
