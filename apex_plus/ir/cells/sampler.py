from apex_plus.ir.cell import Cell


class Sampler(Cell):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
