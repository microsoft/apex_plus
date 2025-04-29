from typing import List

from apex_plus.ir.task import Task


class Cell:

    def __init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    def get_tasks(self) -> List[Task]:
        raise NotImplementedError

    def get_num_task_types(self) -> int:
        raise NotImplementedError

    def has_same_spec(self, other: object) -> bool:
        raise NotImplementedError

    def is_attn(self) -> bool:
        return any(task.is_attn() for task in self.get_tasks())
