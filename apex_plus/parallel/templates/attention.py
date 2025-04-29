from typing import Dict, List, Optional

from apex_plus.ir.cells.attention import MQA
from apex_plus.ir.task import Task
from apex_plus.parallel.comm import CollectiveComm
from apex_plus.parallel.task_parallel import ParallelTemplate, TaskMapping
from apex_plus.parallel.templates.default import DefaultTemplate


class MQATemplate0(ParallelTemplate):

    @staticmethod
    def map_tasks(cell: MQA, num_devices: int) -> Optional[TaskMapping]:
        if cell.num_query_heads < num_devices:
            # Not enough heads to distribute.
            return None

        # Evenly distribute heads to devices, while co-locating the query heads
        # that correspond to the same key and value heads.
        if cell.num_kv_heads < num_devices:
            # KV heads are replicated across devices.
            num_devices_per_kv_head = [
                num_devices // cell.num_kv_heads for _ in range(cell.num_kv_heads)
            ]
            for i in range(num_devices % cell.num_kv_heads):
                num_devices_per_kv_head[i] += 1

            query_heads_per_device: List[Dict[str, List[Task]]] = []
            for i in range(num_devices):
                query_heads_per_device.append({})

            for i in range(cell.num_kv_heads):
                num_devices = num_devices_per_kv_head[i]
                num_query_heads_per_device = [
                    cell.num_query_per_kv // num_devices for _ in range(num_devices)
                ]
                for j in range(cell.num_query_per_kv % num_devices):
                    num_query_heads_per_device[j] += 1

                start = i * cell.num_query_per_kv
                for j in range(num_devices):
                    end = start + num_query_heads_per_device[j]
                    query_heads_per_device[j].setdefault("MQAHead", []).extend(
                        cell.heads[start:end]
                    )
                    start = end
        else:
            # Distibute KV heads to devices as evenly as possible.
            num_kv_heads_per_device = [
                cell.num_kv_heads // num_devices for _ in range(num_devices)
            ]
            for i in range(cell.num_kv_heads % num_devices):
                num_kv_heads_per_device[i] += 1

            query_heads_per_device: List[Dict[str, List[Task]]] = []
            for i in range(num_devices):
                query_heads_per_device.append({})
            start = 0
            for i in range(num_devices):
                end = start + num_kv_heads_per_device[i]
                query_heads_per_device[i].setdefault("MQAHead", []).extend(
                    cell.heads[
                        start * cell.num_query_per_kv : end * cell.num_query_per_kv
                    ]
                )
                start = end

        task_mapping = TaskMapping(
            query_heads_per_device,
            CollectiveComm(comm_type="AllReduce", num_devices=num_devices),
        )
        return task_mapping


# Cell name -> list of templates.
ATTENTION_TEMPLATES_REGISTRY = {
    "MHA": [DefaultTemplate],
    "BiMHA": [DefaultTemplate],
    "MQA": [MQATemplate0],
    "ParallelMHAMLP": [DefaultTemplate],
}
