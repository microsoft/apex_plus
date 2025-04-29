from typing import List

from apex_plus.ir.cell import Cell
from apex_plus.parallel.task_parallel import ParallelTemplate
from apex_plus.parallel.templates.attention import ATTENTION_TEMPLATES_REGISTRY
from apex_plus.parallel.templates.ffn import FFN_TEMPLATES_REGISTRY


def get_templates(cell: Cell) -> List[ParallelTemplate]:
    """Get the list of templates for a cell."""
    cell_name = cell.get_name()
    if cell_name in ATTENTION_TEMPLATES_REGISTRY:
        return ATTENTION_TEMPLATES_REGISTRY[cell_name]
    elif cell_name in FFN_TEMPLATES_REGISTRY:
        return FFN_TEMPLATES_REGISTRY[cell_name]
    else:
        raise ValueError(f"Unknown cell name: {cell_name}")
