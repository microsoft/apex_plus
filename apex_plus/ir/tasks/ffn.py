from typing import List

from apex_plus.ir.task import Task
from apex_plus.utils.dtype import DTYPE


class MLPFilter(Task):

    def __init__(
        self,
        filter_id: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.filter_id = filter_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def __repr__(self) -> str:
        return f"MLPFilter(filter_id={self.filter_id})"

    @staticmethod
    def get_param_size(tasks: List["MLPFilter"], dtype: DTYPE) -> int:
        filter = tasks[0]
        cnt = 2 * filter.hidden_size
        return len(tasks) * cnt * dtype.size


class GLUFilter(Task):

    def __init__(
        self,
        filter_id: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.filter_id = filter_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def __repr__(self) -> str:
        return f"GLUFilter(filter_id={self.filter_id})"

    @staticmethod
    def get_param_size(tasks: List["GLUFilter"], dtype: DTYPE) -> int:
        filter = tasks[0]
        cnt = 3 * filter.hidden_size
        return len(tasks) * cnt * dtype.size


class SwiGLUFilter(Task):

    def __init__(
        self,
        filter_id: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.filter_id = filter_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def __repr__(self) -> str:
        return f"SwiGLUFilter(filter_id={self.filter_id})"

    @staticmethod
    def get_param_size(tasks: List["SwiGLUFilter"], dtype: DTYPE) -> int:
        filter = tasks[0]
        cnt = 3 * filter.hidden_size
        return len(tasks) * cnt * dtype.size


class ExpertMLPFilter(Task):

    def __init__(
        self,
        expert_id: int,
        filter_id: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.expert_id = expert_id
        self.filter_id = filter_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def __repr__(self) -> str:
        return (
            f"ExpertMLPFilter(expert_id={self.expert_id}, "
            f"filter_id={self.filter_id})"
        )

    def get_type(self) -> str:
        return f"ExpertMLPFilter_{self.expert_id}"

    @staticmethod
    def get_param_size(tasks: List["ExpertMLPFilter"], dtype: DTYPE) -> int:
        filter = tasks[0]
        cnt = 2 * filter.hidden_size
        return len(tasks) * cnt * dtype.size


class ExpertSwiGLUFilter(Task):

    def __init__(
        self,
        expert_id: int,
        filter_id: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.expert_id = expert_id
        self.filter_id = filter_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def __repr__(self) -> str:
        return (
            f"ExpertSwiGLUFilter(expert_id={self.expert_id}, "
            f"filter_id={self.filter_id})"
        )

    def get_type(self) -> str:
        return f"ExpertSwiGLUFilter_{self.expert_id}"

    @staticmethod
    def get_param_size(tasks: List["ExpertSwiGLUFilter"], dtype: DTYPE) -> int:
        filter = tasks[0]
        cnt = 3 * filter.hidden_size
        return len(tasks) * cnt * dtype.size
