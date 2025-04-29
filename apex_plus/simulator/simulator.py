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
WORKSPACE = 1 * GB  # a constant buffer for each device to run the program

MAX_NUM_INPUT_TOKENS = 64 * 1024  # Max in profile/scripts/gemm.py

US_TO_SEC = 1000000
MS_TO_SEC = 1000
US_TO_MS = 1000


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
    ) -> None:
        self.model = model
        self.cluster = cluster
        self.trace = trace
        self.dtype = dtype
        self.gpu = cluster.get_device().device_type
        self.gpu_memory = cluster.get_device_memory_capacity()
        self.peak_flops = cluster.get_device().peak_flops[self.highest_prec()]
        self.peak_mem_bandwidth = cluster.get_device().peak_mem_bandwidth
        self.num_total_nodes = cluster.get_num_nodes()
        self.num_total_devices = cluster.get_num_devices()
        self.cluster_size_per_node = self.num_total_devices // self.num_total_nodes

    def highest_prec(self) -> DTYPE:
        data_type = []
        data_type.append(self.dtype["w"])
        data_type.append(self.dtype["kv"])
        data_type.append(self.dtype["act"])
        # Dealing with mixed precesion
        # Assuming we dequantize the value for computation
        highest_precision = DTYPE.FLOAT8
        if DTYPE.FLOAT16 in data_type:
            highest_precision = DTYPE.FLOAT16
        if DTYPE.FLOAT32 in data_type:
            highest_precision = DTYPE.FLOAT32
        return highest_precision

    def dispatch(
        self,
        requests: List[Request],
        factor: int,
    ) -> List[List[Request]]:
        sublists = [[] for _ in range(factor)]
        # Distribute elements in a round-robin fashion
        # Can be replaced with more sophisticated strategy
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

            # Calculate avg TBT per request
            for latency_list in token_latencies_per_request:
                list_length = len(latency_list)
                avg_tbt = 0
                if list_length > 1:
                    ttft = latency_list[0]
                    ttlt = sum(latency_list)
                    avg_tbt = (ttlt - ttft) / list_length
                avg_tbt_vals.append(avg_tbt)

            # Calculate all necessary percentiles for tbt
            for percentile in token_percentiles:
                percentile_vals.append(np.percentile(avg_tbt_vals, percentile))

            return percentile_vals

        def calculate_slo_metrics(latency_dict, slo_targets):
            token_latencies_per_request = [latency for latency in latency_dict.values()]
            num_requests = len(token_latencies_per_request)
            ttft_target = slo_targets[0]
            tpot_target = slo_targets[1]
            slo_metrics = []

            # Calculate Percentage of requests that are <= TTFT_SLO
            # TTFT is just the first token in the latency
            ttft_slo_counter = 0
            for latency_list in token_latencies_per_request:
                if(latency_list[0]/US_TO_MS <= ttft_target):
                    ttft_slo_counter += 1
            slo_metrics.append( (ttft_slo_counter/num_requests) * 100 )

            # Calculate Percentage of requests that have an avg TPOT <= TPOT_SLO
            tpot_slo_counter = 0
            tok_latencies_per_req_after_first_tok = [sublist[1:] for sublist in token_latencies_per_request]
            for latency_list in token_gen_times:
                # Calculate Avg TPOT per request
                avg_tpot = np.mean(latency_list)/US_TO_MS
                if(avg_tpot <= tpot_target):
                    tpot_slo_counter  += 1
            slo_metrics.append( (tpot_slo_counter/num_requests) * 100 )

            return slo_metrics
        
        # Store performance metrics - Time to first token, TPOT ,P50, P95, & other latencies
        performance_metrics: List[Tuple[str, float]] = []
        performance_metrics_units: List[str] = []
        performance_metrics.append(
            ("Throughput: Avg. Tokens generated per second", float("NaN"))
        )
        performance_metrics.append(
            ("Throughput: Avg. Tokens processed per second", float("NaN"))
        )
        performance_metrics.append(("Throughput: Requests per second", float("NaN")))
        performance_metrics.append(
            ("Latency: Avg. Time to first token (TTFT in msec)", float("NaN"))
        )
        performance_metrics.append(
            ("Latency: Avg. Time per output token (TPOT in msec)", float("NaN"))
        )
        performance_metrics_units += [
            "tokens/sec",
            "tokens/sec",
            "requests/sec",
            "msec",
            "msec",
        ]

        num_layers = 0
        num_heads = 0
        head_dim = 0
        hidden_size = 0
        theoretical_peak_flops = self.peak_flops

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
        # If encoder, there the output_len is 0
        if arch == "encoder":
            avg_output_len = 0.0
        # Decoders that generate tokens
        else:
            # TPOT after the first token(this is also known as inter-token latency)
            token_gen_times = [value[1:] for value in request_token_gen_times.values()]
            flat_token_gen_times = [
                item for sublist in token_gen_times for item in sublist
            ]
            tpot = np.mean(flat_token_gen_times) / MS_TO_SEC
            avg_ttft = (
                np.mean([value[0] for value in request_token_gen_times.values()])
                / US_TO_MS
            )

            # Calculate token percentiles
            token_percentiles = token_percentiles + [50, 95]
            token_percentiles.sort()
            # Avg percentile vals are returned in order of sorted percentiles
            avg_percentile_vals = calculate_tbt_percentiles(request_token_gen_times)
            # Add to performance_metrics
            for index, percentile in enumerate(token_percentiles):
                performance_metrics.append(
                    (
                        f"Avg. TBT Percentile: P{percentile}",
                        avg_percentile_vals[index] / US_TO_MS,
                    )
                )
                performance_metrics_units.append("msec")
            # MBU
            kv_cache_size = (
                2 * num_layers * num_heads * head_dim * self.dtype["kv"].size
            )
            tpot_sec = tpot / MS_TO_SEC
            theoretical_peak_mem_bandwidth = self.peak_mem_bandwidth
            observed_mem_bandwidth = (num_parameters + kv_cache_size) / tpot_sec
            mbu = (observed_mem_bandwidth / theoretical_peak_mem_bandwidth) * 100

        # Tokens gen per second
        token_throughput = avg_output_len * len(requests) / (total_time / US_TO_SEC)
        performance_metrics[0] = (performance_metrics[0][0], token_throughput)
        # Tokens processed per second
        performance_metrics[1] = (
            performance_metrics[1][0],
            (avg_input_len + avg_output_len) * len(requests) / (total_time / US_TO_SEC),
        )
        # Requests per second
        performance_metrics[2] = (
            performance_metrics[2][0],
            len(requests) / (total_time / US_TO_SEC),
        )
        # Time to first token
        performance_metrics[3] = (performance_metrics[3][0], avg_ttft)
        # TPOT after the first token(this is also known as inter-token latency)
        performance_metrics[4] = (performance_metrics[4][0], tpot)
        # Calculate request percentiles

        request_latencies = [
            sum(token_latencies) for token_latencies in request_token_gen_times.values()
        ]
        req_percentiles = req_percentiles + [50, 95]
        req_percentiles.sort()
        for percentile in req_percentiles:
            percentile_val = np.percentile(request_latencies, percentile) / US_TO_SEC
            performance_metrics.append(
                (
                    f"Request Completion Latency: {percentile}th percentile",
                    percentile_val,
                )
            )
            performance_metrics_units.append("sec")

        # Calculate Avg MFU
        observed_throughput = token_throughput
        mfus: List[float] = []
        for index, seq_len in enumerate(seq_lens):
            theoretical_throughput = theoretical_peak_flops / (
                6 * num_parameters + 12 * num_layers * num_heads * head_dim * seq_len
            )
            mfu = (observed_throughput / theoretical_throughput) * 100
            mfus.append(mfu)

        avg_mfu = np.mean(mfus)
        performance_metrics.append((f"Avg. MFU Per iteration", avg_mfu))
        performance_metrics_units.append("%")

        # Append mbu
        performance_metrics.append((f"MBU ", mbu))
        performance_metrics_units.append("%")

        # Get SLO Metrics
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
    ) -> Optional[SimulatorOutput]:
        parallel_schedule = execution_plan.parallel_schedule
        stage_schedule = parallel_schedule.stage_schedule
        num_stages = parallel_schedule.num_stages
        num_model_replicas = parallel_schedule.num_model_replicas
        num_attn_cell_replicas = 0
        for cell_schedule in stage_schedule.cell_schedules:
            if cell_schedule.cell.is_attn():
                num_attn_cell_replicas = cell_schedule.num_replicas
                break
        assert num_attn_cell_replicas > 0

        param_sizes = stage_schedule.get_param_size_per_device(self.dtype["w"])
        num_devices = len(param_sizes)

        available_memories = [
            self.gpu_memory - WORKSPACE - param_size for param_size in param_sizes
        ]
        if any(avail_mem < 0 for avail_mem in available_memories):
            # Invalid.
            return None
        min_available_memory = min(available_memories) + WORKSPACE
        param_size = max(param_sizes)

        # Calculate the maximum number of tokens that can be stored in KV cache.
        # This limits the maximum number of sequences that can be batched.
        kv_token_sizes = (
            [1] * num_devices
            if arch == "encoder"
            else stage_schedule.get_kv_token_size_per_device(self.dtype["kv"])
        )

        max_num_tokens = min(
            int(available_memories[i] // kv_token_sizes[i]) for i in range(num_devices)
        )
        # Evenly partition the KV cache for each stage.
        max_num_tokens_per_stage = max_num_tokens // num_stages

        # Statistics
        list_of_exe_time = []
        num_reqs_per_iteration = []
        num_tokens_per_iteration = []
        request_token_gen_times = {}

        requests = []
        model_replica_time = []

        total_energy = 0.0
        # Copy to avoid altering the original traces
        copied_requests = copy.deepcopy(self.trace.requests)

        ### Finished housekeeping; starts the actual simulation ###

        # Split a list of requests to n sublists,
        # where n = num_model_replicas
        model_requests = self.dispatch(copied_requests, num_model_replicas)
        for model_replica in range(num_model_replicas):
            model_replica_energy = 0.0
            stage_iter_times = []
            stage_requests = self.dispatch(model_requests[model_replica], num_stages)
            for stage in range(num_stages):

                cell_replica_iter_times = []
                cell_requests = self.dispatch(
                    stage_requests[stage], num_attn_cell_replicas
                )
                for cell_replica in range(num_attn_cell_replicas):
                    target_requests = cell_requests[cell_replica]

                    (
                        updated_requests,
                        cell_replica_iter_time,
                        reqs_per_iter,
                        tokens_per_iter,
                        exe_time,
                        request_token_gen_time,
                        stage_energy,
                    ) = self.sub_simulate(
                        execution_plan,
                        arch,
                        frequency,
                        target_requests,
                        num_attn_cell_replicas,
                        max_num_tokens_per_stage,
                        max_batch_size,
                    )

                    if updated_requests is None:
                        return None, None
                    requests.extend(
                        updated_requests
                    )  # timestamp updated with completion time
                    id_num = model_replica * stage * cell_replica
                    renamed_gen_time = {
                        f"{key}_{id_num}": value
                        for key, value in request_token_gen_time.items()
                    }
                    request_token_gen_times.update(renamed_gen_time)
                    cell_replica_iter_times.append(cell_replica_iter_time)
                    num_reqs_per_iteration.append(reqs_per_iter)
                    num_tokens_per_iteration.append(tokens_per_iter)
                    list_of_exe_time.append(exe_time)
                    model_replica_energy += (
                        stage_energy // num_attn_cell_replicas * num_stages
                    )
                    # Note: dividied by cell replicas as the energy scaling of cell is already handled in Line 693
                    # Multiplied with num_stages because we only simulate one stage, but num_stages run concurrently

                # iteration time = slowest among the cell replicas
                max_cell_iter_time = self.merge_max_elements(cell_replica_iter_times)
                stage_iter_times.append(max_cell_iter_time)

            interleaved_list = [
                val
                for pair in itertools.zip_longest(*stage_iter_times)
                for val in pair
                if val is not None
            ]

            model_replica_iter_times = []
            if len(interleaved_list) <= num_stages:
                model_replica_iter_times = interleaved_list.copy()
            else:
                # Creates a sliding window to find the bottlenecked stage in the pipeline
                for i in range(len(interleaved_list) - num_stages):
                    window = interleaved_list[i : i + num_stages]
                    model_replica_iter_times.append(max(window))
            model_replica_time.append(sum(model_replica_iter_times))
            total_energy += model_replica_energy

        # Final execution time = the slowest among the replicas
        total_time = max(model_replica_time)

        ### Finished simulation; calculate the statistics of the results ###

        avg_input_len = sum(request.input_len for request in copied_requests) / len(
            copied_requests
        )
        avg_output_len = sum(request.output_len for request in copied_requests) / len(
            copied_requests
        )
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
            slo_targets)

        exe_stat_dict = {}
        idle_time = []
        for exe_t in list_of_exe_time:
            # Calculating idle time
            idle_time.append(total_time - sum(t * num_stages for _, t in exe_t))
            summed_data = defaultdict(float)
            # A function may be called multiple times in one iter, creating multiple entries
            # merging these entries for each iter
            for key, value in exe_t:
                summed_data[key] += value
            result = [(key, value) for key, value in summed_data.items()]
            # Modifying the data structure so it's easier to compute mean and std later
            for name, time in result:
                exe_stat_dict.setdefault(name, []).append(time * num_stages)

        # Compute the statistics
        exe_stat = []
        for name in exe_stat_dict.keys():
            exe_lst = exe_stat_dict.get(name)
            # Wait time will be counted as Idle time
            if name == "Wait":
                for i, val in enumerate(exe_lst):
                    idle_time[i] += val
                continue
            exe_mean = np.mean(exe_lst)
            exe_std = np.std(exe_lst)
            exe_stat.append((name, exe_mean, exe_std))
        exe_stat.append(("Idle", np.mean(idle_time), np.std(idle_time)))

        requests = sorted(requests, key=lambda x: x.time_stamp)

        return requests, SimulatorOutput(
            param_size_per_device=param_size / GB,
            available_memory_per_device=min_available_memory / GB,
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
    ):
        parallel_schedule = execution_plan.parallel_schedule
        stage_schedule = parallel_schedule.stage_schedule
        num_stages = parallel_schedule.num_stages

        min_num_replicas = min(
            cell_schedule.num_replicas
            for cell_schedule in stage_schedule.cell_schedules
        )

        num_cached_tokens = 0
        req_counter = 0
        num_generated_tokens: Dict[int, int] = {}  # request_id -> num_tokens
        running: List[int] = []  # request_ids
        stopped: List[int] = []  # request_ids

        get_seq_len = lambda request_id: (
            requests[request_id].input_len + num_generated_tokens[request_id]
        )

        # Statistics.
        execution_time: List[Tuple[str, float]] = []
        num_cells = len(stage_schedule.cell_schedules)
        for i in range(num_cells):
            cell = stage_schedule.cell_schedules[i].cell
            execution_time.append((cell.get_name(), 0.0))
            for comm in stage_schedule.reshard_comms[i]:
                execution_time.append((comm.comm_type.name, 0.0))
        if parallel_schedule.num_stages > 1:
            execution_time.append(("SendRecv", 0.0))

        num_reqs_per_iteration: List[int] = []
        num_tokens_per_iteration: List[int] = []
        # Simulate the execution.
        time_per_iteration: List[float] = []
        # Time metrics for each request, request id is the key and value of list is time per token
        request_token_gen_times: Dict[str, List[float]] = {}
        internal_clock = 0  # decide whether a request has arrived
        wait_next_req_time = 0  # the idle time of waiting for next request to come
        energy = 0  # energy consumption
        while True:
            # Batch requests.
            input_lens: List[int] = []
            cached_lens: List[int] = []

            new_running: List[int] = []
            while running:
                request_id = running.pop(0)
                while num_cached_tokens + 1 > max_num_tokens_per_stage:
                    if running:
                        victim = running.pop(-1)
                        stopped.append(victim)
                        num_cached_tokens -= get_seq_len(victim)
                    else:
                        stopped.append(request_id)
                        num_cached_tokens -= get_seq_len(request_id)
                        break
                else:
                    input_lens.append(1)
                    num_cached_tokens += 1
                    cached_lens.append(num_generated_tokens[request_id] + 1)
                    new_running.append(request_id)
            running = new_running

            # Resume the stopped requests.
            # Sort in the order of request_id.
            stopped = sorted(stopped)
            while stopped:
                request_id = stopped[0]
                seq_len = get_seq_len(request_id)
                if num_cached_tokens + seq_len + 1 > max_num_tokens_per_stage:
                    break
                request_id = stopped.pop(0)
                input_lens.append(1)
                num_cached_tokens += seq_len + 1
                cached_lens.append(num_generated_tokens[request_id] + 1)
                running.append(request_id)

            # Batch new requests.
            if not stopped:
                while req_counter < len(requests):
                    request_id = req_counter
                    input_len = requests[request_id].input_len
                    # If the KV cache does not have enough space, stop.
                    if num_cached_tokens + input_len > max_num_tokens_per_stage:
                        break

                    num_tokens = sum(input_lens) + input_len
                    # If the total number of tokens exceeds the maximum, stop.
                    if (
                        num_tokens * num_attn_cell_replicas / min_num_replicas
                        > MAX_NUM_INPUT_TOKENS
                    ):
                        break
                    
                    curr_batch_size = len(running)
                    if(curr_batch_size == max_batch_size and max_batch_size != 0):
                        break
                    
                    # Request has not yet arrived
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
                    # Cannot proceed.
                    # This can happen when the space for the KV cache is
                    # too small to store even a single sequence.
                    if num_cached_tokens + input_len > max_num_tokens_per_stage:
                        return None, None, None, None, None, None
                    else:
                        # Or because the requests are coming too slow;
                        # wait until next request comes.
                        if requests[req_counter].time_stamp > internal_clock:
                            wait_next_req_time += (
                                requests[req_counter].time_stamp - internal_clock
                            )
                            internal_clock = requests[req_counter].time_stamp
                        else:
                            return None, None, None, None, None, None

                else:
                    # All the requests are finished.
                    assert num_cached_tokens == 0, num_cached_tokens
                    assert not stopped, stopped
                    break

            # Record the number of requests and tokens.
            num_reqs_per_iteration.append(len(running) * num_attn_cell_replicas)
            num_tokens_per_iteration.append(sum(input_lens) * num_attn_cell_replicas)

            # Get the execution time of a stage with the given input if running
            if running:
                stage_execution_time, stage_energy = self.get_stage_execution_time(
                    execution_plan.parallel_schedule.stage_schedule,
                    num_attn_cell_replicas,
                    input_lens,
                    cached_lens,
                    self.gpu,
                    frequency,
                    self.cluster_size_per_node,
                )
                if num_stages > 1:
                    stage_execution_time.append(
                        self.get_cross_stage_comm_time(
                            sum(input_lens),
                            execution_plan.stage_clusters,
                            self.gpu,
                            self.cluster_size_per_node,
                        )
                    )
                time_per_iteration.append(sum(stage_execution_time))
                internal_clock += sum(stage_execution_time)
                energy += sum(stage_energy)
                # Update the statistics.
                for i in range(len(execution_time)):
                    execution_time[i] = (
                        execution_time[i][0],
                        execution_time[i][1] + stage_execution_time[i],
                    )

                # Remove finished requests from the batch. Update logged time per token
                for request_id in running:
                    num_generated_tokens[request_id] += 1
                    if num_generated_tokens[request_id] == 1:
                        request_token_gen_times[request_id] = [
                            time_per_iteration[-1] * num_stages
                        ]
                    else:
                        request_token_gen_times[request_id].append(
                            time_per_iteration[-1] * num_stages
                        )
                new_running: List[int] = []
                for request_id in running:
                    num_generated = num_generated_tokens[request_id]
                    if arch == "encoder":
                        output_len = 0
                    else:
                        output_len = requests[request_id].output_len
                    if num_generated < output_len:
                        new_running.append(request_id)
                    else:
                        # Finished processing; update the time_stamp to completed time.
                        num_cached_tokens -= get_seq_len(request_id) - 1
                        requests[request_id].time_stamp = internal_clock * num_stages
                running = new_running

        execution_time.append(("Wait", wait_next_req_time))

        return (
            requests,
            time_per_iteration,
            np.mean(num_reqs_per_iteration),
            np.mean(num_tokens_per_iteration),
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
    ) -> List[float]:
        # Calculate the number of input tokens per cell.
        num_total_input_tokens = (
            sum(input_lens_per_attn_replica) * num_attn_cell_replicas
        )

        execution_time: List[float] = []
        execution_energy: List[float] = []
        for i, cell_schedule in enumerate(stage_schedule.cell_schedules):
            # Split the input tokens evenly among the replicas.
            num_replicas = cell_schedule.num_replicas
            num_input_tokens = (
                num_total_input_tokens + num_replicas - 1
            ) // num_replicas
            num_devices = cell_schedule.get_num_devices()
            # For mixed precision, assume the data with lower precision
            # will be dequantized to match data of higher precision
            comp_type = self.highest_prec()

            # Cell execution.
            # We leverage the fact that the 0-th device is always assigned the
            # most number of tasks.
            cell_execution_time = 0.0
            cell_execution_energy = 0.0
            task_dict = cell_schedule.task_mapping.tasks_per_device[0]
            for task_type, tasks in task_dict.items():
                if task_type == "MHAHead" or task_type == "MQAHead":
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
                    # Each expert will get topk / E of the input tokens where E
                    # is the total number of experts.
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

            if (
                cell_schedule.cell.get_name() == "MoE"
                or cell_schedule.cell.get_name() == "SwiMoE"
            ):
                if len(task_dict) < cell_schedule.cell.num_experts:
                    num_devices = len(cell_schedule.task_mapping.tasks_per_device)
                    num_input_tokens = max(num_input_tokens // num_devices, 1)

            hidden_size = self.model.hidden_size
            # Resharding.
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
                    raise NotImplementedError(
                        f"Unsupported comm type: {comm.comm_type}"
                    )
                comm_time = get_comm_time(
                    comm.comm_type,
                    gpu_type,
                    num_nodes,
                    num_devices_per_node,
                    self.dtype["act"],
                    num_elements,
                )
                execution_time.append(comm_time)

        # Multiply the block execution time by the number of blocks.
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
        if cross_node:
            return get_p2p_comm_time(
                gpu=gpu_type,
                num_nodes=2,
                num_gpus_per_node=1,
                dtype=self.dtype["act"],
                num_elements=num_input_tokens * hidden_size,
            )
        else:
            return get_p2p_comm_time(
                gpu=gpu_type,
                num_nodes=1,
                num_gpus_per_node=2,
                dtype=self.dtype["act"],
                num_elements=num_input_tokens * hidden_size,
            )
