from typing import Dict, List, Optional

from apex_plus.ir.cell import Cell
from apex_plus.ir.task import Task
from apex_plus.parallel.comm import CollectiveComm
from apex_plus.utils.dtype import DTYPE


class TaskMapping:

    def __init__(
        self,
        tasks_per_device: List[Dict[str, List[Task]]],  # [device][task_type] -> tasks
        collective_comm: CollectiveComm,
    ) -> None:
        self.tasks_per_device = tasks_per_device
        self.collective_comm = collective_comm

    def __repr__(self) -> str:
        num = len(self.tasks_per_device)
        task_dict = self.tasks_per_device[0]
        task_type, homogeneous_tasks = next(iter(task_dict.items()))
        tasks_per_device = {task_type: len(homogeneous_tasks)}

        return f"TaskMapping(tasks_per_device={tasks_per_device})x{num}"

    def get_num_devices(self) -> int:
        return len(self.tasks_per_device)

    def get_param_sizes(self, dtype: DTYPE) -> List[int]:
        param_size_per_device: List[int] = []
        for task_dict in self.tasks_per_device:
            param_size = 0
            for homogeneous_tasks in task_dict.values():
                task = homogeneous_tasks[0]
                param_size += task.get_param_size(homogeneous_tasks, dtype)
            param_size_per_device.append(param_size)
        return param_size_per_device

    def get_kv_token_sizes(self, dtype: DTYPE) -> List[int]:
        kv_token_size_per_device: List[int] = []
        for task_dict in self.tasks_per_device:
            kv_token_size = 0
            for homogeneous_tasks in task_dict.values():
                task = homogeneous_tasks[0]
                if not task.is_attn():
                    continue
                kv_token_size += task.get_kv_token_size(homogeneous_tasks, dtype)
            kv_token_size_per_device.append(kv_token_size)
        return kv_token_size_per_device


class ParallelTemplate:

    @staticmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        raise NotImplementedError
