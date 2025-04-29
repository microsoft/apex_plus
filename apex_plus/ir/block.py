from typing import List

from apex_plus.ir.cell import Cell


class Block:

    def __init__(
        self,
        cells: List[Cell],
    ) -> None:
        self.cells = cells
