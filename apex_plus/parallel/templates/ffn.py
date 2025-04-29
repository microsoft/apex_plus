from typing import Dict, List, Optional

from apex_plus.ir.cells.ffn import MoE
from apex_plus.ir.task import Task
from apex_plus.parallel.comm import CollectiveComm
from apex_plus.parallel.task_parallel import ParallelTemplate, TaskMapping
from apex_plus.parallel.templates.default import DefaultTemplate


class MoETemplate0(ParallelTemplate):

    @staticmethod
    def map_tasks(cell: MoE, num_devices: int) -> Optional[TaskMapping]:
        # Distribute the experts evenly across devices.
        if cell.num_experts < num_devices:
            # Not enough number of experts.
            return None

        num_experts_per_device = [
            cell.num_experts // num_devices for _ in range(num_devices)
        ]
        for i in range(cell.num_experts % num_devices):
            num_experts_per_device[i] += 1
        intermediate_size = cell.intermediate_size

        tasks_per_device: List[Dict[str, List[Task]]] = []
        start = 0
        for i in range(num_devices):
            tasks_per_device.append({})
            end = start + num_experts_per_device[i]
            for expert_id in range(start, end):
                expert = cell.expert_filters[expert_id * intermediate_size]
                task_type = expert.get_type()
                tasks_per_device[i][task_type] = cell.expert_filters[
                    expert_id * intermediate_size : (expert_id + 1) * intermediate_size
                ]
            start = end

        task_mapping = TaskMapping(
            tasks_per_device,
            CollectiveComm(comm_type="AllGather", num_devices=num_devices),
        )
        return task_mapping


# Cell name -> list of templates.
FFN_TEMPLATES_REGISTRY = {
    "SwiGLU": [DefaultTemplate],
    "GLU": [DefaultTemplate],
    "MLP": [DefaultTemplate],
    "MoE": [DefaultTemplate, MoETemplate0],
    "SwiMoE": [DefaultTemplate, MoETemplate0],
}
