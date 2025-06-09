from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import itertools
import copy

from apex_plus.cluster.cluster import Cluster
from apex_plus.execution.plan import ExecutionPlan
from apex_plus.ir.transformer import Transformer
from apex_plus.parallel.comm import CommType
from apex_plus.parallel.schedule import StageSchedule
from apex_plus.simulator.comm_profile import get_comm_time, get_p2p_comm_time
from apex_plus.simulator.comp_profile import mha_time, mlp_time, glu_time, swiglu_time
from apex_plus.simulator.trace import Trace, Request
from apex_plus.utils.dtype import DTYPE

GB = 1024 * 1024 * 1024
WORKSPACE = 1 * GB
MAX_NUM_INPUT_TOKENS = 64 * 1024
US_TO_SEC = 1000000
MS_TO_SEC = 1000
US_TO_MS = 1000
KV_CACHE_TRANSFER_POWER = 1000

@dataclass
class SimulatorOutput:
    param_size_per_device: float
    available_memory_per_device: float
    num_requests_per_iteration: float
    num_tokens_per_iteration: float
    time_statistics: List[Tuple[str, float]]
    performance_metrics: List[Tuple[str, float]]
    performance_metrics_units: List[str]
    slo_metrics: List[float]
    total_time: float
    total_energy: float

class Simulator:
    def __init__(
        self,
        model: Transformer,
        cluster: Cluster,
        trace: Trace,
        dtype: dict,
        prefill_gpu: int = 0,  # GPU for prefill phase
        decode_gpu: int = 1,   # GPU for decode phase
    ) -> None:
        self.model = model
        self.cluster = cluster
        self.trace = trace
        self.dtype = dtype
        self.prefill_gpu = prefill_gpu
        self.decode_gpu = decode_gpu
        self.gpu = cluster.get_device().device_type
        self.gpu_memory = cluster.get_device_memory_capacity()
        self.peak_flops = cluster.get_device().peak_flops[self.highest_prec()]
        self.peak_mem_bandwidth = cluster.get_device().peak_mem_bandwidth
        self.num_total_nodes = cluster.get_num_nodes()
        self.num_total_devices = cluster.get_num_devices()
        self.cluster_size_per_node = self.num_total_devices // self.num_total_nodes

    def highest_prec(self) -> DTYPE:
        data_type = [self.dtype["w"], self.dtype["kv"], self.dtype["act"]]
        highest_precision = DTYPE.FLOAT8
        if DTYPE.FLOAT16 in data_type:
            highest_precision = DTYPE.FLOAT16
        if DTYPE.FLOAT32 in data_type:
            highest_precision = DTYPE.FLOAT32
        return highest_precision

    def dispatch(self, requests: List[Request], factor: int) -> List[List[Request]]:
        sublists = [[] for _ in range(factor)]
        for index, element in enumerate(requests):
            sublists[index % factor].append(element)
        return sublists

    def merge_max_elements(self, lists):
        max_length = max(len(lst) for lst in lists)
        extended_lists = [lst + [None] * (max_length - len(lst)) for lst in lists]
        merged_list = [
            max(filter(lambda x: x is not None, elements))
            for elements in zip(*extended_lists)
        ]
        return merged_list

    def get_metrics(
        self,
        model_config: Transformer,
        avg_input_len: int = 0,
        avg_output_len: int = 0,
        requests: List[Trace] = [],
        total_time: float = 0.0,
        request_token_gen_times: Dict[str, List[float]] = {},
        req_percentiles: List[int] = [],
        token_percentiles: List[int] = [],
        seq_lens: List[int] = [],
        arch: str = '',
        slo_targets: List[int] = [],
    ):
        def calculate_tbt_percentiles(latency_dict):
            token_latencies_per_request = [latency for latency in latency_dict.values()]
            avg_tbt_vals = []
            percentile_vals = []
            for latency_list in token_latencies_per_request:
                list_length = len(latency_list)
                avg_tbt = 0
                if list_length > 1:
                    ttft = latency_list[0]
                    ttlt = sum(latency_list)
                    avg_tbt = (ttlt - ttft) / list_length
                avg_tbt_vals.append(avg_tbt)
            for percentile in token_percentiles:
                percentile_vals.append(np.percentile(avg_tbt_vals, percentile))
            return percentile_vals

        def calculate_slo_metrics(latency_dict, slo_targets):
            token_latencies_per_request = [latency for latency in latency_dict.values()]
            num_requests = len(token_latencies_per_request)
            ttft_target = slo_targets[0]
            tpot_target = slo_targets[1]
            slo_metrics = []
            ttft_slo_counter = 0
            for latency_list in token_latencies_per_request:
                if latency_list[0] / US_TO_MS <= ttft_target:
                    ttft_slo_counter += 1
            slo_metrics.append((ttft_slo_counter / num_requests) * 100)
            tpot_slo_counter = 0
            for latency_list in token_latencies_per_request:
                avg_tpot = np.mean(latency_list[1:]) / US_TO_MS if len(latency_list) > 1 else float('inf')
                if avg_tpot <= tpot_target:
                    tpot_slo_counter += 1
            slo_metrics.append((tpot_slo_counter / num_requests) * 100)
            return slo_metrics

        performance_metrics: List[Tuple[str, float]] = []
        performance_metrics_units: List[str] = []
        performance_metrics.append(("Throughput: Avg. Tokens generated per second", float("NaN")))
        performance_metrics.append(("Throughput: Avg. Tokens processed per second", float("NaN")))
        performance_metrics.append(("Throughput: Requests per second", float("NaN")))
        performance_metrics.append(("Latency: Avg. Time to first token (TTFT in msec)", float("NaN")))
        performance_metrics.append(("Latency: Avg. Time per output token (TPOT in msec)", float("NaN")))
        performance_metrics_units += ["tokens/sec", "tokens/sec", "requests/sec", "msec", "msec"]

        if hasattr(model_config, "num_layers"):
            num_layers = model_config.num_layers
            num_heads = model_config.num_heads
            hidden_size = model_config.hidden_size
        elif hasattr(model_config, "num_decoder_layers"):
            num_layers = model_config.num_decoder_layers
            num_heads = model_config.num_decoder_heads
            hidden_size = model_config.decoder_hidden_size
        else:
            raise ValueError("Unable to get model layers, heads, or hidden size")
        head_dim = hidden_size // num_heads
        num_parameters = num_layers * hidden_size * hidden_size * 12

        tpot = 0.0
        avg_ttft = 0.0
        mbu = 0.0
        if arch != "encoder":
            token_gen_times = [value[1:] for value in request_token_gen_times.values()]
            flat_token_gen_times = [item for sublist in token_gen_times for item in sublist]
            tpot = np.mean(flat_token_gen_times) / MS_TO_SEC if flat_token_gen_times else 0.0
            avg_ttft = np.mean([value[0] for value in request_token_gen_times.values()]) / US_TO_MS
            token_percentiles = sorted(token_percentiles + [50, 95])
            avg_percentile_vals = calculate_tbt_percentiles(request_token_gen_times)
            for index, percentile in enumerate(token_percentiles):
                performance_metrics.append((f"Avg. TBT Percentile: P{percentile}", avg_percentile_vals[index] / US_TO_MS))
                performance_metrics_units.append("msec")
            kv_cache_size = 2 * num_layers * num_heads * head_dim * self.dtype["kv"].size
            tpot_sec = tpot / MS_TO_SEC
            observed_mem_bandwidth = (num_parameters + kv_cache_size) / tpot_sec if tpot_sec else 0
            mbu = (observed_mem_bandwidth / self.peak_mem_bandwidth) * 100

        token_throughput = avg_output_len * len(requests) / (total_time / US_TO_SEC) if total_time else 0
        performance_metrics[0] = (performance_metrics[0][0], token_throughput)
        performance_metrics[1] = (performance_metrics[1][0], (avg_input_len + avg_output_len) * len(requests) / (total_time / US_TO_SEC))
        performance_metrics[2] = (performance_metrics[2][0], len(requests) / (total_time / US_TO_SEC))
        performance_metrics[3] = (performance_metrics[3][0], avg_ttft)
        performance_metrics[4] = (performance_metrics[4][0], tpot)

        request_latencies = [sum(token_latencies) for token_latencies in request_token_gen_times.values()]
        req_percentiles = sorted(req_percentiles + [50, 95])
        for percentile in req_percentiles:
            percentile_val = np.percentile(request_latencies, percentile) / US_TO_SEC
            performance_metrics.append((f"Request Completion Latency: {percentile}th percentile", percentile_val))
            performance_metrics_units.append("sec")

        observed_throughput = token_throughput
        mfus: List[float] = []
        for seq_len in seq_lens:
            theoretical_throughput = self.peak_flops / (6 * num_parameters + 12 * num_layers * num_heads * head_dim * seq_len)
            mfu = (observed_throughput / theoretical_throughput) * 100
            mfus.append(mfu)
        avg_mfu = np.mean(mfus)
        performance_metrics.append((f"Avg. MFU Per iteration", avg_mfu))
        performance_metrics_units.append("%")
        performance_metrics.append((f"MBU", mbu))
        performance_metrics_units.append("%")

        slo_metrics = calculate_slo_metrics(request_token_gen_times, slo_targets)
        return performance_metrics, performance_metrics_units, slo_metrics

    def simulate(
        self,
        execution_plan: ExecutionPlan,
        arch: str,
        frequency: int,
        model_config: Transformer,
        req_percentiles: List[int] = [],
        token_percentiles: List[int] = [],
        slo_targets: List[int] = [],
        max_batch_size: int = 0,
        distserve: bool = False,
    ) -> Optional[SimulatorOutput]:
        if self.num_total_devices < 2:
            raise ValueError("Need at least 2 GPUs for prefill and decode disaggregation")

        parallel_schedule = execution_plan.parallel_schedule
        stage_schedule = parallel_schedule.stage_schedule
        num_stages = parallel_schedule.num_stages
        num_model_replicas = parallel_schedule.num_model_replicas
        num_attn_cell_replicas = next(
            cell_schedule.num_replicas for cell_schedule in stage_schedule.cell_schedules if cell_schedule.cell.is_attn()
        )
        # print(f"cluster: {self.cluster}")
        param_sizes = stage_schedule.get_param_size_per_device(self.dtype["w"])
        if len(param_sizes) == 1:
            return None, None
        # print(f"param_sizes: {param_sizes}")
        # print(f"self.prefill_gpu: {self.prefill_gpu}")
        # print(f"self.decode_gpu: {self.decode_gpu}")
        available_memory_prefill = self.gpu_memory - WORKSPACE - param_sizes[self.prefill_gpu]
        available_memory_decode = self.gpu_memory - WORKSPACE - param_sizes[self.decode_gpu]
        if any(avail_mem < 0 for avail_mem in [available_memory_prefill, available_memory_decode]):
            return None
        min_available_memory_prefill = available_memory_prefill + WORKSPACE
        min_available_memory_decode = available_memory_decode + WORKSPACE
        param_size = max(param_sizes)

        # KV cache on decode GPU only
        kv_token_sizes = [0] * self.num_total_devices
        kv_token_sizes[self.decode_gpu] = stage_schedule.get_kv_token_size_per_device(self.dtype["kv"])[0]
        max_num_tokens = int(available_memory_decode // kv_token_sizes[self.decode_gpu])
        max_num_tokens_per_stage = max_num_tokens // num_stages

        list_of_exe_time = []
        num_reqs_per_iteration = []
        num_tokens_per_iteration = []
        request_token_gen_times = {}
        total_energy = 0.0
        copied_requests = copy.deepcopy(self.trace.requests)

        # Split requests across model replicas
        model_requests = self.dispatch(copied_requests, num_model_replicas)
        total_time = 0.0

        for model_replica in range(num_model_replicas):
            model_replica_energy = 0.0
            model_replica_time = []

            # Prefill phase on prefill GPU
            prefill_requests = copy.deepcopy(model_requests[model_replica])
            (
                prefill_updated_requests,
                prefill_iter_times,
                prefill_reqs_per_iter,
                prefill_tokens_per_iter,
                prefill_exe_time,
                prefill_token_gen_times,
                prefill_energy,
            ) = self.sub_simulate(
                execution_plan,
                arch,
                frequency,
                prefill_requests,
                num_attn_cell_replicas,
                max_num_tokens_per_stage,
                max_batch_size,
                phase="prefill",
            )
            if prefill_updated_requests is None:
                return None

            # Calculate KV cache transfer time
            kv_cache_size = sum(
                2 * model_config.num_layers * model_config.num_heads * (model_config.hidden_size // model_config.num_heads) * self.dtype["kv"].size * req.input_len
                for req in prefill_updated_requests
            )
            if self.num_total_nodes == 1:
                kv_transfer_time = get_p2p_comm_time(
                    gpu=self.gpu,
                    num_nodes=1,
                    num_gpus_per_node=2,
                    dtype=self.dtype["kv"],
                    num_elements=kv_cache_size,
                )
            else:
                kv_transfer_time = get_p2p_comm_time(
                    gpu=self.gpu,
                    num_nodes=2,
                    num_gpus_per_node=1,
                    dtype=self.dtype["kv"],
                    num_elements=kv_cache_size,
                )


            # Decode phase on decode GPU
            decode_requests = prefill_updated_requests
            (
                decode_updated_requests,
                decode_iter_times,
                decode_reqs_per_iter,
                decode_tokens_per_iter,
                decode_exe_time,
                decode_token_gen_times,
                decode_energy,
            ) = self.sub_simulate(
                execution_plan,
                arch,
                frequency,
                decode_requests,
                num_attn_cell_replicas,
                max_num_tokens_per_stage,
                max_batch_size,
                phase="decode",
            )
            if decode_updated_requests is None:
                return None

            # Combine statistics
            renamed_gen_times = {
                f"{key}_{model_replica}": prefill_token_gen_times.get(key, []) + decode_token_gen_times.get(key, [])
                for key in set(list(prefill_token_gen_times.keys()) + list(decode_token_gen_times.keys()))
            }
            request_token_gen_times.update(renamed_gen_times)
            # list_of_exe_time.extend(prefill_exe_time + [(name, t + kv_transfer_time) if name != "Wait" else (name, t) for name, t in decode_exe_time])
            # integrate the prefill and decode exe time
            prefill_exe_time_dict = {name: t for name, t in prefill_exe_time}
            decode_exe_time_dict = {name: t for name, t in decode_exe_time}
            total_exe_time = []
            for name, t in decode_exe_time_dict.items():
                if name in prefill_exe_time_dict:
                    total_exe_time.append((name, prefill_exe_time_dict[name] + decode_exe_time_dict[name]))
            total_exe_time.append(("KV Transfer", kv_transfer_time))
            list_of_exe_time.append(total_exe_time)
            num_reqs_per_iteration.extend(prefill_reqs_per_iter + decode_reqs_per_iter)
            num_tokens_per_iteration.extend(prefill_tokens_per_iter + decode_tokens_per_iter)
            model_replica_energy += prefill_energy + decode_energy + kv_transfer_time * KV_CACHE_TRANSFER_POWER
            model_replica_time.extend(prefill_iter_times + decode_iter_times + [kv_transfer_time])
            total_energy += model_replica_energy
            total_time = max(total_time, sum(model_replica_time))

        avg_input_len = sum(request.input_len for request in copied_requests) / len(copied_requests)
        avg_output_len = sum(request.output_len for request in copied_requests) / len(copied_requests)
        performance_metrics, performance_metrics_units, slo_metrics = self.get_metrics(
            model_config,
            avg_input_len,
            avg_output_len,
            copied_requests,
            total_time,
            request_token_gen_times,
            req_percentiles,
            token_percentiles,
            num_tokens_per_iteration,
            arch,
            slo_targets,
        )

        exe_stat_dict = {}
        idle_time = []
        # print(f"list_of_exe_time: {list_of_exe_time}")  
        for exe_t in list_of_exe_time:
            # print(f"exe_t: {exe_t}")
            idle_time.append(total_time - sum(t * num_stages for _, t in exe_t))
            summed_data = defaultdict(float)
            for key, value in exe_t:
                summed_data[key] += value
            result = [(key, value) for key, value in summed_data.items()]
            for name, time in result:
                exe_stat_dict.setdefault(name, []).append(time * num_stages)
        exe_stat = []
        for name in exe_stat_dict.keys():
            exe_lst = exe_stat_dict.get(name)
            if name == "Wait":
                for i, val in enumerate(exe_lst):
                    idle_time[i] += val
                continue
            exe_mean = np.mean(exe_lst)
            exe_std = np.std(exe_lst)
            exe_stat.append((name, exe_mean, exe_std))
        exe_stat.append(("Idle", np.mean(idle_time), np.std(idle_time)))

        requests = sorted(copied_requests, key=lambda x: x.time_stamp)
        return requests, SimulatorOutput(
            param_size_per_device=param_size / GB,
            available_memory_per_device=min_available_memory_decode / GB,
            num_requests_per_iteration=np.mean(num_reqs_per_iteration),
            num_tokens_per_iteration=np.mean(num_tokens_per_iteration),
            time_statistics=exe_stat,
            performance_metrics=performance_metrics,
            performance_metrics_units=performance_metrics_units,
            slo_metrics=slo_metrics,
            total_time=total_time,
            total_energy=total_energy,
        )

    def sub_simulate(
        self,
        execution_plan: ExecutionPlan,
        arch: str,
        frequency: int,
        requests: List[Request],
        num_attn_cell_replicas: int,
        max_num_tokens_per_stage: int,
        max_batch_size: int,
        phase: str = "combined",  # "prefill" or "decode"
    ):
        parallel_schedule = execution_plan.parallel_schedule
        stage_schedule = parallel_schedule.stage_schedule
        num_stages = parallel_schedule.num_stages
        min_num_replicas = min(cell_schedule.num_replicas for cell_schedule in stage_schedule.cell_schedules)

        num_cached_tokens = 0
        req_counter = 0
        num_generated_tokens: Dict[int, int] = {}
        running: List[int] = []
        stopped: List[int] = []
        get_seq_len = lambda request_id: (
            requests[request_id].input_len + num_generated_tokens.get(request_id, 0)
        )

        execution_time: List[Tuple[str, float]] = []
        for i, cell_schedule in enumerate(stage_schedule.cell_schedules):
            cell = cell_schedule.cell
            execution_time.append((cell.get_name(), 0.0))
            for comm in stage_schedule.reshard_comms[i]:
                execution_time.append((comm.comm_type.name, 0.0))
        if num_stages > 1:
            execution_time.append(("SendRecv", 0.0))

        num_reqs_per_iteration: List[int] = []
        num_tokens_per_iteration: List[int] = []
        time_per_iteration: List[float] = []
        request_token_gen_times: Dict[str, List[float]] = {}
        internal_clock = 0
        wait_next_req_time = 0
        energy = 0

        while True:
            input_lens: List[int] = []
            cached_lens: List[int] = []
            new_running: List[int] = []

            if phase == "prefill":
                # Batch requests for prefill
                while running:
                    request_id = running.pop(0)
                    seq_len = get_seq_len(request_id)
                    if num_cached_tokens + seq_len > max_num_tokens_per_stage:
                        stopped.append(request_id)
                        num_cached_tokens -= seq_len
                    else:
                        input_lens.append(requests[request_id].input_len)
                        cached_lens.append(0)
                        new_running.append(request_id)
                running = new_running

                # Add new requests
                while req_counter < len(requests):
                    request_id = req_counter
                    input_len = requests[request_id].input_len
                    if num_cached_tokens + input_len > max_num_tokens_per_stage:
                        break
                    num_tokens = sum(input_lens) + input_len
                    if num_tokens * num_attn_cell_replicas / min_num_replicas > MAX_NUM_INPUT_TOKENS:
                        break
                    if len(running) == max_batch_size and max_batch_size != 0:
                        break
                    if requests[request_id].time_stamp > internal_clock:
                        break
                    num_cached_tokens += input_len
                    input_lens.append(input_len)
                    cached_lens.append(0)
                    running.append(request_id)
                    num_generated_tokens[request_id] = 0
                    req_counter += 1

                if not running:
                    if req_counter < len(requests):
                        if num_cached_tokens + input_len > max_num_tokens_per_stage:
                            return None, None, None, None, None, None, None
                        wait_next_req_time += requests[req_counter].time_stamp - internal_clock
                        internal_clock = requests[req_counter].time_stamp
                    else:
                        break

                # Execute prefill
                if running:
                    stage_execution_time, stage_energy = self.get_stage_execution_time(
                        stage_schedule,
                        num_attn_cell_replicas,
                        input_lens,
                        cached_lens,
                        self.gpu,
                        frequency,
                        self.cluster_size_per_node,
                        phase="prefill",
                        gpu_id=self.prefill_gpu,
                    )
                    time_per_iteration.append(sum(stage_execution_time))
                    internal_clock += sum(stage_execution_time)
                    energy += sum(stage_energy)
                    for i in range(len(execution_time)):
                        execution_time[i] = (
                            execution_time[i][0],
                            execution_time[i][1] + stage_execution_time[i],
                        )

                    # Record TTFT
                    for request_id in running:
                        request_token_gen_times[str(request_id)] = [time_per_iteration[-1] * num_stages]
                        requests[request_id].time_stamp = internal_clock * num_stages
                    running = []  # Prefill complete, move to decode

            elif phase == "decode":
                # Batch requests for decode
                while running:
                    request_id = running.pop(0)
                    seq_len = get_seq_len(request_id)
                    if num_cached_tokens + seq_len + 1 > max_num_tokens_per_stage:
                        stopped.append(request_id)
                        num_cached_tokens -= seq_len
                    else:
                        input_lens.append(1)
                        num_cached_tokens += 1
                        cached_lens.append(num_generated_tokens.get(request_id, 0) + 1)
                        new_running.append(request_id)
                running = new_running

                # Resume stopped requests
                stopped = sorted(stopped)
                while stopped:
                    request_id = stopped[0]
                    seq_len = get_seq_len(request_id)
                    if num_cached_tokens + seq_len + 1 > max_num_tokens_per_stage:
                        break
                    request_id = stopped.pop(0)
                    input_lens.append(1)
                    num_cached_tokens += seq_len + 1
                    cached_lens.append(num_generated_tokens.get(request_id, 0) + 1)
                    running.append(request_id)

                # Add new requests
                while req_counter < len(requests):
                    request_id = req_counter
                    input_len = requests[request_id].input_len
                    if num_cached_tokens + input_len + 1 > max_num_tokens_per_stage:
                        break
                    num_tokens = sum(input_lens) + 1
                    if num_tokens * num_attn_cell_replicas / min_num_replicas > MAX_NUM_INPUT_TOKENS:
                        break
                    if len(running) >= max_batch_size and max_batch_size != 0:
                        break
                    if requests[request_id].time_stamp > internal_clock:
                        break
                    num_cached_tokens += input_len + 1
                    input_lens.append(1)
                    cached_lens.append(num_generated_tokens.get(request_id, 0) + 1)
                    running.append(request_id)
                    num_generated_tokens[request_id] = 0
                    req_counter += 1

                if not running:
                    if req_counter < len(requests):
                        if num_cached_tokens + input_len > max_num_tokens_per_stage:
                            return None, None, None, None, None, None, None
                        wait_next_req_time += requests[req_counter].time_stamp - internal_clock
                        internal_clock = requests[req_counter].time_stamp
                    else:
                        break

                # Execute decode
                if running:
                    stage_execution_time, stage_energy = self.get_stage_execution_time(
                        stage_schedule,
                        num_attn_cell_replicas,
                        input_lens,
                        cached_lens,
                        self.gpu,
                        frequency,
                        self.cluster_size_per_node,
                        phase="decode",
                        gpu_id=self.decode_gpu,
                    )
                    time_per_iteration.append(sum(stage_execution_time))
                    internal_clock += sum(stage_execution_time)
                    energy += sum(stage_energy)
                    for i in range(len(execution_time)):
                        execution_time[i] = (
                            execution_time[i][0],
                            execution_time[i][1] + stage_execution_time[i],
                        )

                    # Update token generation
                    for request_id in running:
                        num_generated_tokens[request_id] = num_generated_tokens.get(request_id, 0) + 1
                        if str(request_id) not in request_token_gen_times:
                            request_token_gen_times[str(request_id)] = []
                        request_token_gen_times[str(request_id)].append(time_per_iteration[-1] * num_stages)

                    # Remove finished requests
                    new_running = []
                    for request_id in running:
                        if num_generated_tokens[request_id] < requests[request_id].output_len:
                            new_running.append(request_id)
                        else:
                            num_cached_tokens -= get_seq_len(request_id) - 1
                            requests[request_id].time_stamp = internal_clock * num_stages
                    running = new_running

            else:
                raise ValueError(f"Unsupported phase: {phase}")

            num_reqs_per_iteration.append(len(running) * num_attn_cell_replicas)
            num_tokens_per_iteration.append(sum(input_lens) * num_attn_cell_replicas)

        execution_time.append(("Wait", wait_next_req_time))
        return (
            requests,
            time_per_iteration,
            num_reqs_per_iteration,
            num_tokens_per_iteration,
            execution_time,
            request_token_gen_times,
            energy,
        )

    def get_stage_execution_time(
        self,
        stage_schedule: StageSchedule,
        num_attn_cell_replicas: int,
        input_lens_per_attn_replica: List[int],
        cached_lens_per_attn_replica: List[int],
        gpu_type: str,
        frequency: int,
        cluster_size_per_node: int,
        phase: str = "combined",
        gpu_id: int = 0,
    ) -> Tuple[List[float], List[float]]:
        num_total_input_tokens = sum(input_lens_per_attn_replica) * num_attn_cell_replicas
        execution_time: List[float] = []
        execution_energy: List[float] = []
        for i, cell_schedule in enumerate(stage_schedule.cell_schedules):
            num_replicas = cell_schedule.num_replicas
            num_input_tokens = (num_total_input_tokens + num_replicas - 1) // num_replicas
            num_devices = cell_schedule.get_num_devices()
            comp_type = self.highest_prec()

            cell_execution_time = 0.0
            cell_execution_energy = 0.0
            task_dict = cell_schedule.task_mapping.tasks_per_device[0]
            for task_type, tasks in task_dict.items():
                if task_type in ["MHAHead", "MQAHead"]:
                    exe_time, exe_energy = mha_time(
                        gpu_type,
                        frequency,
                        tasks,
                        comp_type,
                        input_lens_per_attn_replica,
                        cached_lens_per_attn_replica,
                        True,
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                elif task_type == "BiMHAHead":
                    exe_time, exe_energy = mha_time(
                        gpu_type,
                        frequency,
                        tasks,
                        comp_type,
                        input_lens_per_attn_replica,
                        cached_lens_per_attn_replica,
                        False,
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                elif task_type == "MLPFilter":
                    exe_time, exe_energy = mlp_time(
                        gpu_type, frequency, tasks, comp_type, num_input_tokens
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                elif task_type == "GLUFilter":
                    exe_time, exe_energy = glu_time(
                        gpu_type, frequency, tasks, comp_type, num_input_tokens
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                elif task_type == "SwiGLUFilter":
                    exe_time, exe_energy = swiglu_time(
                        gpu_type, frequency, tasks, comp_type, num_input_tokens
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                elif task_type.startswith("ExpertMLPFilter"):
                    num_total_experts = cell_schedule.cell.num_experts
                    topk = cell_schedule.cell.topk
                    exe_time, exe_energy = mlp_time(
                        gpu_type,
                        frequency,
                        tasks,
                        comp_type,
                        max(num_input_tokens * topk // num_total_experts, 1),
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                elif task_type.startswith("ExpertSwiGLUFilter"):
                    num_total_experts = cell_schedule.cell.num_experts
                    topk = cell_schedule.cell.topk
                    exe_time, exe_energy = swiglu_time(
                        gpu_type,
                        frequency,
                        tasks,
                        comp_type,
                        max(num_input_tokens * topk // num_total_experts, 1),
                    )
                    cell_execution_time += exe_time
                    cell_execution_energy += exe_energy
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
            execution_time.append(cell_execution_time)
            execution_energy.append(cell_execution_energy * num_devices)

            if cell_schedule.cell.get_name() in ["MoE", "SwiMoE"]:
                if len(task_dict) < cell_schedule.cell.num_experts:
                    num_devices = len(cell_schedule.task_mapping.tasks_per_device)
                    num_input_tokens = max(num_input_tokens // num_devices, 1)

            hidden_size = self.model.hidden_size
            for comm in stage_schedule.reshard_comms[i]:
                if comm.num_devices < cluster_size_per_node:
                    num_nodes = 1
                    num_devices_per_node = comm.num_devices
                else:
                    num_nodes = comm.num_devices // cluster_size_per_node
                    num_devices_per_node = cluster_size_per_node
                num_input_tokens *= comm.size_factor
                num_input_tokens = max(num_input_tokens, 1)
                if comm.comm_type == CommType.AllReduce:
                    num_elements = num_input_tokens * hidden_size
                elif comm.comm_type == CommType.AllGather:
                    num_elements = num_input_tokens * comm.num_devices * hidden_size
                    num_input_tokens *= comm.num_devices
                elif comm.comm_type == CommType.ReduceScatter:
                    num_elements = num_input_tokens * hidden_size
                    num_input_tokens = max(num_input_tokens // comm.num_devices, 1)
                elif comm.comm_type == CommType.AllToAll:
                    num_elements = num_input_tokens * hidden_size
                else:
                    raise NotImplementedError(f"Unsupported comm type: {comm.comm_type}")
                comm_time = get_comm_time(
                    comm.comm_type,
                    gpu_type,
                    num_nodes,
                    num_devices_per_node,
                    self.dtype["act"],
                    num_elements,
                )
                execution_time.append(comm_time)

        return [t * stage_schedule.num_blocks for t in execution_time], [
            e * stage_schedule.num_blocks for e in execution_energy
        ]

    def get_cross_stage_comm_time(
        self,
        num_input_tokens: int,
        stage_clusters: List[Cluster],
        gpu_type: str,
        cluster_size_per_node: int,
    ) -> float:
        hidden_size = self.model.hidden_size
        num_total_devices = sum(cluster.get_num_devices() for cluster in stage_clusters)
        cross_node = num_total_devices > cluster_size_per_node
        return get_p2p_comm_time(
            gpu=gpu_type,
            num_nodes=2 if cross_node else 1,
            num_gpus_per_node=1,
            dtype=self.dtype["act"],
            num_elements=num_input_tokens * hidden_size,
        )