from typing import Dict, List, Optional

from apex_plus.ir.cell import Cell
from apex_plus.ir.task import Task
from apex_plus.parallel.comm import CollectiveComm
from apex_plus.parallel.task_parallel import ParallelTemplate, TaskMapping


class DefaultTemplate(ParallelTemplate):

    @staticmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        tasks_per_type: Dict[str, List[Task]] = {}
        for task in cell.get_tasks():
            task_type = task.get_type()
            if task_type not in tasks_per_type:
                tasks_per_type[task_type] = []
            tasks_per_type[task_type].append(task)

        tasks_per_device: List[Dict[str, List[Task]]] = []
        for i in range(num_devices):
            tasks_per_device.append({})

        # For each task type, distribute tasks to devices.
        for task_type, tasks in tasks_per_type.items():
            # Distribute tasks as evenly as possible.
            num_tasks = len(tasks)
            num_tasks_per_device = [num_tasks // num_devices] * num_devices
            for i in range(num_tasks % num_devices):
                num_tasks_per_device[i] += 1

            start = 0
            for i in range(num_devices):
                end = start + num_tasks_per_device[i]
                tasks_per_device[i][task_type] = tasks[start:end]
                start = end

        if all(len(x) == 0 for x in tasks_per_device[-1].values()):
            # Not enough tasks to distribute.
            return None

        # Create a task mapping with AllReduce.
        task_mapping = TaskMapping(
            tasks_per_device,
            CollectiveComm(comm_type="AllReduce", num_devices=num_devices),
        )
        return task_mapping
