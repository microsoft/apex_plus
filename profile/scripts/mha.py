import argparse

import pandas as pd
import ray
import torch
from torch.profiler import profile, ProfilerActivity
from xformers import ops as xops

import pynvml
import subprocess
import os
import signal
import threading
import queue
import time

NUM_WARMUP = 5

H: Number of attention heads up to 128.
H = [1, 2, 4, 8, 16, 24, 32, 40, 48, 52, 64, 96, 128]
# D: Head dimension.
D = [64, 80, 96, 128, 112, 256]
# B: Batch size up to 4096
B = list(range(1, 32)) + [1 << i for i in range(5, 13)]
# L: Sequence length up to 16K.
L = [16 * i for i in range(1, 1025)]
# B * L <= MAX_NUM_TOKENS
MAX_NUM_TOKENS = 16 * 4096



@ray.remote(num_gpus=1)
class Profiler:

    def __init__(self, idx: int, num_gpus: int):
        self.idx = idx
        self.num_gpus = num_gpus

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def _get_gpu_freq_pairs(self):
        max_pairs = []
        memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.handle)
        try:
            for mem_clk in memory_clocks:
                graphics_clks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.handle, mem_clk)
                if graphics_clks:
                    max_graphics_clk = max(graphics_clks)
                    max_pairs.append((mem_clk, max_graphics_clk))
        except:
            print(f"Error when getting valid freq pairs")
            exit(1)
        return [(2619, 1980), (2619, 810)]
 
    def _set_gpu_freq(self, mem_clk, graph_clk):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-ac", f"{mem_clk},{graph_clk}"],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Exit code: {e.returncode}")
            print(f"Stderr: {e.stderr.strip()}")
            exit(1)

    def _reset_gpu_freq(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-rac"],
                check=True,
                capture_output=True,
                text=True
            )
            print("Clocks reset to default")
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print("Failed to reset clocks")
            print("Exit code:", e.returncode)
            print("Error:", e.stderr.strip()) 

    def _start_power_logging(self, log_file: str, tag: str, interval_ms: int = 50):
        stop_event = threading.Event()
        q = queue.Queue()

        def logger():
            with open(log_file, "a") as f:
                while not stop_event.is_set():
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True
                    )
                    power = result.stdout.strip().split("\n")[0]
                    timestamp = time.time()
                    f.write(f"{power},{timestamp},{tag}\n")
                    f.flush()
                    time.sleep(interval_ms / 1000.0)

        t = threading.Thread(target=logger)
        t.start()
        return stop_event, t
    
    def _stop_power_logging(self, stop_event, thread):
        stop_event.set()
        thread.join()

    def _get_avg_power(self, log_file: str, tag: str) -> float:
        try:
            with open(log_file) as f:
                readings = [
                    float(line.split(",")[0])
                    for line in f if line.strip() and line.strip().endswith(tag)
                ]
            return sum(readings) / len(readings) if readings else 0.0
        except Exception as e:
            print(f"Error reading power log for tag={tag}: {e}")
            return 0.0


    def _profile(self, h: int, d: int, b: int, l: int, dtype: str, attention: str) -> float:
        if dtype == "half":
            dtype = torch.float16
        elif dtype == "float":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        q = torch.randn(b, l, h, d, device="cuda", dtype=dtype)
        k = torch.randn(b, l, h, d, device="cuda", dtype=dtype)
        v = torch.randn(b, l, h, d, device="cuda", dtype=dtype)

        tag = f"h{h}_d{d}_b{b}_l{l}"
        power_log_file = f"{attention}_power_log_{self.idx}.csv"
        stop_event, thread = self._start_power_logging(power_log_file, tag)

        # Use CUTLASS backend.
        xops_backend = xops.fmha.cutlass.FwOp()
        if attention == "MHA":
            xops_causal_mask = xops.LowerTriangularMask()
        else:
            xops_causal_mask = None
        scale = d ** -0.5

        # Warmup
        for _ in range(NUM_WARMUP):
            xops.memory_efficient_attention_forward(q, k, v,
                                                    attn_bias=xops_causal_mask,
                                                    scale=scale, p=0.0,
                                                    op=xops_backend)
        torch.cuda.synchronize()

        # Measure the kernel execution time.
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            xops.memory_efficient_attention_forward(q, k, v,
                                                    attn_bias=xops_causal_mask,
                                                    scale=scale, p=0.0,
                                                    op=xops_backend)
        time.sleep(0.05)
        self._stop_power_logging(stop_event, thread)
        time.sleep(0.05)

        avg_power = self._get_avg_power(power_log_file, tag)
        stats = prof.key_averages()
        return sum(s.device_time for s in stats), avg_power

    def profile(
        self,
        gpu: str,
        dtype: str,
        attention: str,
    ) -> pd.DataFrame:
        data = []
        freq_pairs = self._get_gpu_freq_pairs()
        for mem_clk, graph_clk in freq_pairs:
            print(f"Changing frequency to {mem_clk},{graph_clk} ")
            try:
                self._set_gpu_freq(mem_clk, graph_clk)
                for hi, h in enumerate(H):
                    for di, d in enumerate(D):
                        for bi, b in enumerate(B):
                            for li, l in enumerate(L):
                                if b * l > MAX_NUM_TOKENS:
                                    continue
                                i = (hi * len(D) * len(B) * len(L) + di * len(B) * len(
                                    L) + bi * len(L) + li)
                                if i % self.num_gpus != self.idx:
                                    continue
                                t, avg_power = self._profile(h, d, b, l, dtype, attention)
                                avg_energy = int(t) * avg_power
                                data.append((gpu, dtype, h, d, b, l, int(t),  mem_clk, graph_clk, avg_power, avg_energy))
            except Exception as e:
                print(f"Error when profiling at frequency of {mem_clk}, {graph_clk}: {e}")
                continue
        self._reset_gpu_freq()
        pynvml.nvmlShutdown()
        df = pd.DataFrame(
            data, columns=["gpu", "dtype", "num_heads", "head_size",
                        "batch_size", "seq_len", "time(us)", "mem_clk_freq",
                        "graph_clk_freq", "avg_power(W)", "avg_energy(uJ)"])
        return df


def main(gpu: str, num_gpus: int, dtype: str, attention: str):
    for filename in os.listdir():
        if filename.startswith(f"{attention}_power_log_") and filename.endswith(".csv"):
            os.remove(filename)

    profilers = [Profiler.remote(i, num_gpus) for i in range(num_gpus)]
    profiled_data = []
    for i in range(num_gpus):
        profiler = profilers[i]
        data = profiler.profile.remote(gpu, dtype, attention)
        profiled_data.append(data)
    profiled_data = ray.get(profiled_data)
    df = pd.concat(profiled_data, ignore_index=True)
    df = df.sort_values(by=["gpu", "dtype", "num_heads", "head_size",
                            "batch_size", "seq_len"])
    if attention == "MHA":
        df.to_csv("mha.csv", index=False)
    else:
        df.to_csv("bimha.csv", index=False)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, required=True,
                        choices=["V100-PCIE-16GB","H100-SXM-80GB"])
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="half",
                        choices=["half", "float"])
    parser.add_argument("--attention", type=str, required=True,
                        choices=["MHA", "BiMHA"])
    args = parser.parse_args()

    print(args)
    main(args.gpu, args.num_gpus, args.dtype, args.attention)
