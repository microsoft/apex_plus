from transformers import PretrainedConfig

from apex_plus.ir.transformer import Transformer


class ApexModel:

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> "ApexModel":
        raise NotImplementedError

    def to_ir(self) -> Transformer:
        raise NotImplementedError
