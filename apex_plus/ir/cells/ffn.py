from typing import List

from apex_plus.ir.cell import Cell
from apex_plus.ir.tasks.ffn import (
    ExpertMLPFilter,
    GLUFilter,
    SwiGLUFilter,
    MLPFilter,
    ExpertSwiGLUFilter,
)


class MLP(Cell):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.filters = [
            MLPFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.intermediate_size)
        ]

    def get_tasks(self) -> List[MLPFilter]:
        return self.filters

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MLP):
            return False
        return (
            self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )


class GLU(Cell):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.filters = [
            GLUFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.intermediate_size)
        ]

    def get_tasks(self) -> List[GLUFilter]:
        return self.filters

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, GLU):
            return False
        return (
            self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )


class SwiGLU(Cell):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.filters = [
            SwiGLUFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.intermediate_size)
        ]

    def get_tasks(self) -> List[SwiGLUFilter]:
        return self.filters

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, SwiGLU):
            return False
        return (
            self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )


class MoE(Cell):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        capacity_factor: float = 1.0,
    ) -> None:
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.topk = topk
        self.capacity_factor = capacity_factor

        self.expert_filters: List[ExpertMLPFilter] = []
        for i in range(self.num_experts):
            for j in range(self.intermediate_size):
                expert_filter = ExpertMLPFilter(
                    i, j, self.hidden_size, self.intermediate_size
                )
                self.expert_filters.append(expert_filter)

    def get_tasks(self) -> List[ExpertMLPFilter]:
        return self.expert_filters

    def get_num_task_types(self) -> int:
        return self.num_experts

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MoE):
            return False
        return (
            self.num_experts == other.num_experts
            and self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )


class SwiMoE(Cell):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        capacity_factor: float = 1.0,
    ) -> None:
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.topk = topk
        self.capacity_factor = capacity_factor

        self.expert_filters: List[ExpertSwiGLUFilter] = []
        for i in range(self.num_experts):
            for j in range(self.intermediate_size):
                expert_filter = ExpertSwiGLUFilter(
                    i, j, self.hidden_size, self.intermediate_size
                )
                self.expert_filters.append(expert_filter)

    def get_tasks(self) -> List[ExpertSwiGLUFilter]:
        return self.expert_filters

    def get_num_task_types(self) -> int:
        return self.num_experts

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, SwiMoE):
            return False
        return (
            self.num_experts == other.num_experts
            and self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )
