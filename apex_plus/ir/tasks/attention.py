from typing import List

from apex_plus.ir.task import Task
from apex_plus.utils.dtype import DTYPE


class MHAHead(Task):

    def __init__(
        self,
        head_id: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.head_id = head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    def __repr__(self) -> str:
        return f"MHAHead(head_id={self.head_id})"

    @staticmethod
    def get_param_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        head = tasks[0]
        cnt = 4 * head.hidden_size * head.head_size
        return len(tasks) * cnt * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        head = tasks[0]
        num_heads = len(tasks)
        return 2 * num_heads * head.head_size * dtype.size

    @classmethod
    def is_attn(cls) -> bool:
        return True


class BiMHAHead(Task):

    def __init__(
        self,
        head_id: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.head_id = head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    def __repr__(self) -> str:
        return f"MHAHead(head_id={self.head_id})"

    @staticmethod
    def get_param_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        head = tasks[0]
        cnt = 4 * head.hidden_size * head.head_size
        return len(tasks) * cnt * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MHAHead"], dtype: DTYPE) -> int:
        # Assume no kv cache for bidirectional MHA as it's not used in decoders
        return 0

    @classmethod
    def is_attn(cls) -> bool:
        return True


class MQAHead(Task):

    def __init__(
        self,
        query_head_id: int,
        kv_head_id: int,
        head_size: int,
        hidden_size: int,
    ) -> None:
        self.query_head_id = query_head_id
        self.kv_head_id = kv_head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    def __repr__(self) -> str:
        return (
            f"MQAHead(query_head_id={self.query_head_id}, "
            f"kv_head_id={self.kv_head_id})"
        )

    @staticmethod
    def get_param_size(tasks: List["MQAHead"], dtype: DTYPE) -> int:
        num_query_heads = len(tasks)
        # Different query heads might share the same KV heads.
        num_kv_heads = len(set(task.kv_head_id for task in tasks))
        head_size = tasks[0].head_size
        hidden_size = tasks[0].hidden_size

        q = num_query_heads * head_size * hidden_size
        k = num_kv_heads * head_size * hidden_size
        v = num_kv_heads * head_size * hidden_size
        o = num_query_heads * head_size * hidden_size
        cnt = q + k + v + o
        return cnt * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["MQAHead"], dtype: DTYPE) -> int:
        num_kv_heads = len(set(task.kv_head_id for task in tasks))
        head_size = tasks[0].head_size
        return 2 * num_kv_heads * head_size * dtype.size

    @classmethod
    def is_attn(cls) -> bool:
        return True
