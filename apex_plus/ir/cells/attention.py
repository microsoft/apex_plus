from typing import List, Union

from apex_plus.ir.cell import Cell
from apex_plus.ir.tasks.attention import MHAHead, MQAHead, BiMHAHead
from apex_plus.ir.tasks.ffn import MLPFilter


class MHA(Cell):  # Masked, unidirectional MHA used in decoders

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.heads = [
            MHAHead(i, self.head_size, self.hidden_size) for i in range(self.num_heads)
        ]

    def get_tasks(self) -> List[MHAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MHA):
            return False
        return (
            self.num_heads == other.num_heads and self.hidden_size == other.hidden_size
        )


class BiMHA(Cell):  # Unmasked, bidirectional MHA used in encoders

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.heads = [
            BiMHAHead(i, self.head_size, self.hidden_size)
            for i in range(self.num_heads)
        ]

    def get_tasks(self) -> List[BiMHAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, BiMHAHead):
            return False
        return (
            self.num_heads == other.num_heads and self.hidden_size == other.hidden_size
        )


class ParallelMHAMLP(Cell):

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.heads = [
            MHAHead(i, self.head_size, self.hidden_size) for i in range(self.num_heads)
        ]
        self.filters = [
            MLPFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.hidden_size)
        ]

    def get_tasks(self) -> List[Union[MHAHead, MLPFilter]]:
        return self.heads + self.filters

    def get_num_task_types(self) -> int:
        return 2

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, ParallelMHAMLP):
            return False
        return (
            self.num_heads == other.num_heads
            and self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )


class MQA(Cell):

    def __init__(
        self,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size = hidden_size

        assert self.num_query_heads % self.num_kv_heads == 0
        self.num_query_per_kv = self.num_query_heads // self.num_kv_heads
        self.heads = [
            MQAHead(
                i,
                i // self.num_query_per_kv,
                self.head_size,
                self.hidden_size,
            )
            for i in range(self.num_query_heads)
        ]

    def get_tasks(self) -> List[MQAHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MQA):
            return False
        return (
            self.num_query_heads == other.num_query_heads
            and self.num_kv_heads == other.num_kv_heads
            and self.head_size == other.head_size
            and self.hidden_size == other.hidden_size
        )


class ParallelMQAMLP(Cell):

    def __init__(
        self,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        assert self.num_query_heads % self.num_kv_heads == 0
        self.num_query_per_kv = self.num_query_heads // self.num_kv_heads
        self.heads = [
            MQAHead(
                i,
                i // self.num_query_per_kv,
                self.head_size,
                self.hidden_size,
            )
            for i in range(self.num_query_heads)
        ]
        self.filters = [
            MLPFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.hidden_size)
        ]

    def get_tasks(self) -> List[Union[MQAHead, MLPFilter]]:
        return self.heads + self.filters

    def get_num_task_types(self) -> int:
        return 2

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, ParallelMQAMLP):
            return False
        return (
            self.num_query_heads == other.num_query_heads
            and self.num_kv_heads == other.num_kv_heads
            and self.head_size == other.head_size
            and self.hidden_size == other.hidden_size
            and self.intermediate_size == other.intermediate_size
        )
