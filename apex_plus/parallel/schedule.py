from typing import List

from apex_plus.ir.block import Block
from apex_plus.ir.cell import Cell
from apex_plus.parallel.comm import CollectiveComm
from apex_plus.parallel.task_parallel import TaskMapping
from apex_plus.utils.dtype import DTYPE


class CellSchedule:

    def __init__(
        self,
        cell: Cell,
        num_replicas: int,
        task_mapping: TaskMapping,
    ) -> None:
        self.cell = cell
        self.num_replicas = num_replicas
        self.task_mapping = task_mapping

    def __repr__(self) -> str:
        return (
            f"CellSchedule(num_replicas={self.num_replicas}, "
            f"task_mapping={self.task_mapping})"
        )

    def get_param_sizes(self, dtype: DTYPE) -> List[int]:
        return self.task_mapping.get_param_sizes(dtype) * self.num_replicas

    def get_kv_token_sizes(self, dtype: DTYPE) -> List[int]:
        return self.task_mapping.get_kv_token_sizes(dtype) * self.num_replicas

    def get_num_devices(self) -> int:
        num_devices_per_replica = self.task_mapping.get_num_devices()
        return self.num_replicas * num_devices_per_replica


class StageSchedule:

    def __init__(
        self,
        block: Block,
        num_blocks: int,
        cell_schedules: List[CellSchedule],
        reshard_comms: List[List[CollectiveComm]],
    ) -> None:
        self.block = block
        self.num_blocks = num_blocks
        self.cell_schedules = cell_schedules
        self.reshard_comms = reshard_comms

        num_cells_per_block = len(block.cells)
        assert len(cell_schedules) == num_cells_per_block
        assert len(reshard_comms) == num_cells_per_block

    def get_param_size_per_device(self, dtype: DTYPE) -> List[int]:
        num_devices = self.cell_schedules[0].get_num_devices()
        device_memory_usage = [0] * num_devices
        for cell_schedule in self.cell_schedules:
            param_sizes_per_device = cell_schedule.get_param_sizes(dtype)
            for i in range(num_devices):
                device_memory_usage[i] += param_sizes_per_device[i]
        # Same number of parameters for each block.
        device_memory_usage = [usage * self.num_blocks for usage in device_memory_usage]
        return device_memory_usage

    def get_kv_token_size_per_device(self, dtype: DTYPE) -> List[int]:
        num_devices = self.cell_schedules[0].get_num_devices()
        total_per_device = [0] * num_devices
        for cell_schedule in self.cell_schedules:
            if not cell_schedule.cell.is_attn():
                continue
            kv_token_sizes_per_device = cell_schedule.get_kv_token_sizes(dtype)
            for i in range(num_devices):
                total_per_device[i] += kv_token_sizes_per_device[i]
        # Repeated for each block.
        total_per_device = [size * self.num_blocks for size in total_per_device]
        return total_per_device

    def __repr__(self) -> str:
        msg = f"StageSchedule(num_blocks={self.num_blocks})\n"
        for i in range(len(self.block.cells)):
            cell_schedule = self.cell_schedules[i]
            reshard_comms = self.reshard_comms[i]
            msg += f"  cell_schedule[{i}]: {cell_schedule}\n"
            msg += f"  reshard_comms[{i}]: {reshard_comms}\n"
        return msg


class ParallelSchedule:

    def __init__(
        self,
        num_model_replicas: int,
        num_stages: int,
        stage_schedule: StageSchedule,
    ) -> None:
        self.num_model_replicas = num_model_replicas
        self.num_stages = num_stages
        self.stage_schedule = stage_schedule

    def __repr__(self) -> str:
        return (
            "ParallelSchedule("
            f"num_model_replicas={self.num_model_replicas}, "
            f"num_stages={self.num_stages}, "
            f"stage_schedule={self.stage_schedule})"
        )
