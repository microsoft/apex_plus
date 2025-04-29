import math
import typing
from typing import List, Optional, Tuple

import prettytable
from tqdm import tqdm

from apex_plus.execution.plan import ExecutionPlan
from apex_plus.parallel.reshard import get_reshard_comm, is_reshardable
from apex_plus.parallel.schedule import CellSchedule, ParallelSchedule, StageSchedule
from apex_plus.parallel.templates import get_templates
from apex_plus.simulator.simulator import Simulator, SimulatorOutput
from apex_plus.utils.dtype import DTYPE

if typing.TYPE_CHECKING:
    from apex_plus.cluster.cluster import Cluster
    from apex_plus.ir.block import Block
    from apex_plus.ir.transformer import Transformer
    from apex_plus.parallel.comm import CollectiveComm
    from apex_plus.simulator.trace import Trace


class SearchEngine:

    def __init__(
        self,
        model: "Transformer",
        cluster: "Cluster",
        trace: "Trace",
        arch: str,
        dtype: dict,
    ) -> None:
        self.model = model
        self.cluster = cluster
        self.trace = trace
        self.dtype = dtype
        self.arch = arch

        self.simulator = Simulator(model, cluster, trace, dtype)

    def generate_schedules(
        self,
        num_blocks: int,
        block: "Block",
        cluster: "Cluster",
    ) -> List[ParallelSchedule]:
        parallel_schedules: List[ParallelSchedule] = []

        device_memory_capacity = cluster.get_device_memory_capacity()
        num_devices = cluster.get_num_devices()
        # 1. Model-level data parallelism.
        for num_replicas in _get_divisors(num_devices):
            if not cluster.is_partitionable(num_replicas):
                # Invalid.
                continue

            num_replica_devices = num_devices // num_replicas
            # 2. Pipeline parallelism.
            for num_stages in _get_divisors(num_replica_devices):
                if num_blocks % num_stages != 0:
                    # Cannot evenly distribute blocks.
                    # TODO: Support uneven distribution.
                    continue
                if not cluster.is_partitionable(num_replicas * num_stages):
                    # Invalid.
                    continue

                num_stage_devices = num_replica_devices // num_stages
                num_blocks_per_stage = num_blocks // num_stages

                # Cell index -> possible cell schedules.
                schedules_per_cell: List[List[CellSchedule]] = []
                for cell in block.cells:
                    # Append a new cell.
                    schedules_per_cell.append([])
                    # 3. Cell-level data parallelism.
                    for num_cell_replicas in _get_divisors(num_stage_devices):
                        if not cluster.is_partitionable(
                            num_replicas * num_stages * num_cell_replicas
                        ):
                            # Invalid.
                            continue

                        num_cell_replica_devices = (
                            num_stage_devices // num_cell_replicas
                        )
                        # 4. Task parallelism.
                        for template in get_templates(cell):
                            task_mapping = template.map_tasks(
                                cell, num_cell_replica_devices
                            )
                            if task_mapping is None:
                                # Invalid (e.g, when num_tasks > num_devices).
                                continue
                            cell_schedule = CellSchedule(
                                cell, num_cell_replicas, task_mapping
                            )
                            schedules_per_cell[-1].append(cell_schedule)

                            if num_cell_replica_devices == 1:
                                # No need to consider other task mappings.
                                break

                # Generate all stage schedules.
                num_schedules = 1
                for schedules in schedules_per_cell:
                    num_schedules *= len(schedules)
                for i in range(num_schedules):
                    cell_schedules: List[CellSchedule] = []
                    for schedules in schedules_per_cell:
                        cell_schedules.append(schedules[i % len(schedules)])
                        i //= len(schedules)

                    num_cell_replicas = [
                        cell_schedule.num_replicas for cell_schedule in cell_schedules
                    ]
                    if len(num_cell_replicas) > 1:
                        if math.gcd(*num_cell_replicas) != 1:
                            # For a model replica, the number of cell replicas
                            # must be coprime.
                            continue

                    num_attn_cell_replicas = [
                        cell_schedule.num_replicas
                        for cell_schedule in cell_schedules
                        if cell_schedule.cell.is_attn()
                    ]
                    if len(set(num_attn_cell_replicas)) > 1:
                        # For each attention cell, the number of cell replicas
                        # must be the same. Otherwise, input routing becomes
                        # too complicated.
                        continue

                    # Reshard ops.
                    num_cells_per_block = len(block.cells)
                    reshard_comms: List[Optional["CollectiveComm"]] = []
                    for i in range(num_cells_per_block):
                        c1 = cell_schedules[i]
                        c2 = cell_schedules[(i + 1) % num_cells_per_block]
                        if not is_reshardable(c1, c2):
                            # Unsupported combination of cell schedules. Skip.
                            break
                        comm = get_reshard_comm(c1, c2)
                        reshard_comms.append(comm)
                    if len(reshard_comms) != num_cells_per_block:
                        # Unsupported combination of cell schedules. Skip.
                        continue

                    stage_schedule = StageSchedule(
                        block,
                        num_blocks_per_stage,
                        cell_schedules,
                        reshard_comms,
                    )
                    # Optimization: skip if the schedule is obviously infeasible.
                    # For example, if the number of parameters each device stores
                    # is larger than the memory capacity of the device, then the
                    # schedule is infeasible.
                    param_size_per_device = stage_schedule.get_param_size_per_device(
                        self.dtype["w"]
                    )
                    is_oom = any(
                        param_size > device_memory_capacity
                        for param_size in param_size_per_device
                    )
                    if is_oom:
                        # Infeasible schedule. Skip.
                        continue

                    parallel_schedules.append(
                        ParallelSchedule(
                            num_replicas,
                            num_stages,
                            stage_schedule,
                        )
                    )
        return parallel_schedules

    def generate_plans(self, arch: str, cluster: "Cluster") -> List[ExecutionPlan]:

        # Generate all possible parallel schedules.
        if arch == "encoder":
            parallel_schedules = self.generate_schedules(
                self.model.num_encoder_blocks,
                self.model.encoder_block,
                cluster,
            )
        else:  # decoder
            parallel_schedules = self.generate_schedules(
                self.model.num_decoder_blocks,
                self.model.decoder_block,
                cluster,
            )

        # Generate execution plans by mapping devices.
        execution_plans: List[ExecutionPlan] = []
        for parallel_schedule in parallel_schedules:
            replica_clusters = cluster.partition(parallel_schedule.num_model_replicas)
            replica_cluster = replica_clusters[0]

            stage_clusters = replica_cluster.partition(parallel_schedule.num_stages)
            stage_cluster = stage_clusters[0]

            stage_schedule = parallel_schedule.stage_schedule
            cell_clusters: List[List["Cluster"]] = []
            for cell_schedule in stage_schedule.cell_schedules:
                cell_clusters.append(
                    stage_cluster.partition(cell_schedule.num_replicas)
                )

            execution_plan = ExecutionPlan(
                parallel_schedule,
                stage_clusters,
                cell_clusters,
            )
            execution_plans.append(execution_plan)
        return execution_plans

    def search(
        self,
        return_all_plans: bool,
        frequency: int,
        req_percentiles=[],
        token_percentiles=[],
        model_config=[],
        ttft_slo = 10, 
        tpot_slo = 10, 
        max_batch_size = 0) -> List[ExecutionPlan]:
        """Search for the best execution plan."""
        candidate_plans = self.generate_plans(self.arch, self.cluster)
        print(f"Generated {len(candidate_plans)} {self.arch} candidate plans.")

        outputs: List[Tuple[ExecutionPlan, SimulatorOutput]] = []
        slo_targets = [ttft_slo, tpot_slo]
        for plan in tqdm(candidate_plans):
            requests, output = self.simulator.simulate(
                plan,
                self.arch,
                frequency,
                model_config,
                req_percentiles,
                token_percentiles,
                slo_targets, 
                max_batch_size)
            if output is None:
                # Invalid plan (e.g., when the model does not fit in memory).
                continue
            outputs.append((plan, output))

        if not outputs:
            raise RuntimeError("No valid execution plan found.")

        print("=" * 80)

        outputs = sorted(outputs, key=lambda x: x[1].total_time)
        # Print either best plan or all plans based off flag
        if return_all_plans:
            for i, (plan, output) in enumerate(outputs):
                print(f"* Parallel schedule {i} for {self.arch}:")
                self._print_plan(plan)
                self._print_output(output, ttft_slo, tpot_slo, max_batch_size)
                print("=" * 80)
            return outputs, requests
        else:
            print(f"* Optimal schedule for {self.arch}:")
            best_plan, output = outputs[0]
            self._print_plan(best_plan)
            self._print_output(output, ttft_slo, tpot_slo, max_batch_size)
            print("=" * 80)
            return [best_plan], requests

    def _print_plan(self, plan: ExecutionPlan) -> None:
        # Print the parallel schedule.
        parallel_schedule = plan.parallel_schedule
        stage_schedule = parallel_schedule.stage_schedule
        print(f"  # Model replicas: {parallel_schedule.num_model_replicas}")
        print(f"  # Stages: {parallel_schedule.num_stages}")
        print(f"  # Blocks per stage: {stage_schedule.num_blocks}")

        table = prettytable.PrettyTable(
            align="l",
            border=True,
            hrules=prettytable.FRAME,
            vrules=prettytable.NONE,
            field_names=["Name", "Value"],
        )
        table.max_width["Value"] = 64
        num_cells = len(stage_schedule.block.cells)
        for i in range(num_cells):
            cell = stage_schedule.block.cells[i]
            cell_schedule = stage_schedule.cell_schedules[i]
            table.add_row(
                [
                    cell.get_name(),
                    f"{cell_schedule.num_replicas} replicas, "
                    f"{cell_schedule.task_mapping}",
                ]
            )
            reshard_comms = stage_schedule.reshard_comms[i]
            for comm in reshard_comms:
                table.add_row([comm.comm_type.name, f"{comm.num_devices} devices"])
        print(table)

    def _print_output(self, output: SimulatorOutput, ttft_slo: int, tpot_slo: int, max_batch_size: int) -> None:
        # Print memory and batch statistics.
        table = prettytable.PrettyTable(
            align="l",
            border=True,
            hrules=prettytable.FRAME,
            vrules=prettytable.NONE,
            field_names=["Name", "Value"],
        )
        table.add_row(
            ["Parameter size per device (GB)", f"{output.param_size_per_device:.1f}"]
        )
        table.add_row(
            [
                "Activation memory per device (GB)",
                f"{output.available_memory_per_device:.1f}",
            ]
        )
        table.add_row(
            [
                "Avg. requests per iteration (per microbatch)",
                f"{output.num_requests_per_iteration:.1f}",
            ]
        )
        table.add_row(
            [
                "Avg. tokens per iteration (per microbatch)",
                f"{output.num_tokens_per_iteration:.1f}",
            ]
        )
        print("* Statistics:")
        print(table)

        # Print Performance metrics
        table = prettytable.PrettyTable(
            align="l",
            border=True,
            hrules=prettytable.FRAME,
            vrules=prettytable.NONE,
            field_names=["Name", "Value", "Units"],
        )
        for index, (name, t) in enumerate(output.performance_metrics):
            # Skip the time that is zero.
            if t <= 1e-5:
                continue
            table.add_row([name, f"{t:.3f}", output.performance_metrics_units[index]])

        print("* Performance Metrics:")
        print(table)

        # Print SLO
        slo_targets = [ttft_slo, tpot_slo]
        slo_lables = ["TTFT", "TPOT"]
        table = prettytable.PrettyTable(
            align="l",
            border=True,
            hrules=prettytable.FRAME,
            vrules=prettytable.NONE,
            field_names=["Name", "Target SLO(msec)","SLOs Met(%)"],
        ) 
        for index,t in enumerate(output.slo_metrics):
            # Skip the time that is zero.
            # if t <= 1e-5:
            #     continue
            table.add_row([slo_lables[index], slo_targets[index], f"{t:.3f} %"])
        print("* Latency SLO Metrics:")
        print(f"* Batch Size: {max_batch_size}")
        print(table)


        # Print the time breakdown.
        table = prettytable.PrettyTable(
            align="l",
            border=True,
            hrules=prettytable.FRAME,
            vrules=prettytable.NONE,
            field_names=["Name", "Time (sec)", "Ratio (%)"],
        )
        for name, t, std in output.time_statistics:
            if t <= 1e-5:
                # Skip the time that is zero.
                continue
            table.add_row(
                [
                    name,
                    f"{t / 1000000:.2f} ± {std / 1000000:.1f}",
                    f"{t / output.total_time * 100:.1f}",
                ]
            )
        # Add a horizontal line.
        table.add_row(["Total", f"{output.total_time / 1000000:.2f}", "100.0"])
        print("* Time breakdown:")
        print(table)
        print("Energy Consumption:", f"{output.total_energy / 1000000000:.2f}", "KJ")


def _get_divisors(n: int) -> List[int]:
    """Get all divisors of n."""
    return [i for i in range(1, n + 1) if n % i == 0]
