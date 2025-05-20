import argparse
import time
from typing import List, Tuple

import pandas as pd
import ray
import torch

import subprocess
import os
import threading
import queue


NUM_WARMUP = 5
NUM_ITER = 100

# M: Hidden/intermediate dimension up to 64K.
M = [1 << i for i in range(14 + 1)]
# K: Hidden/intermediate dimension up to 64K.
K = [1 << i for i in range(14 + 1)]
# N: Number of tokens up to 64K
N = list(range(1, 128)) + [128 * i for i in range(1, 513)]


@ray.remote(num_gpus=1)
class GemmProfiler:

    def __init__(self, idx: int, num_gpus: int):
        self.idx = idx
        self.num_gpus = num_gpus
        self.data = []
        runtime_context = ray.get_runtime_context()
        self.worker_id = runtime_context.get_worker_id()
        print(f"Worker ID: {self.worker_id}")

    def get_data(self):
        return self.data

    def _profile(self, m: int, k: int, n: int, dtype: str) -> Tuple[float, float]:
        if dtype == "half":
            dtype = torch.float16
        elif dtype == "float":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        x = torch.randn(m, k, device="cuda", dtype=dtype)
        y = torch.randn(k, n, device="cuda", dtype=dtype)

        tag = f"m{m}_k{k}_n{n}"
        power_log_file = f"power_log_{self.worker_id}.csv"
        stop_event, thread = self._start_power_logging(power_log_file, tag)

        # Warmup
        for _ in range(NUM_WARMUP):
            torch.matmul(x, y)
        torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(NUM_ITER):
            torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()

        time.sleep(0.1)
        self._stop_power_logging(stop_event, thread)
        time.sleep(0.1)

        avg_power = self._get_avg_power(power_log_file, tag)
        avg_time = (end - start) / NUM_ITER
        return avg_time * 1000 * 1000, avg_power

    def _start_power_logging(self, log_file: str, tag: str, interval_ms: int = 50):
        stop_event = threading.Event()
        q = queue.Queue()

        def logger():
            with open(log_file, "a") as f:
                while not stop_event.is_set():
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
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
                    for line in f
                    if line.strip() and line.strip().endswith(tag)
                ]
            return sum(readings) / len(readings) if readings else 0.0
        except Exception as e:
            print(f"Error reading power log for tag={tag}: {e}")
            return 0.0

    def profile(
        self,
        gpu: str,
        dtype: str,
    ) -> pd.DataFrame:
        data: List[Tuple[str, str, int, int, int, int]] = []
        for mi, m in enumerate(M):
            for ki, k in enumerate(K):
                for ni, n in enumerate(N):
                    i = mi * len(K) * len(N) + ki * len(N) + ni
                    if i % self.num_gpus != self.idx:
                        continue
                    t, avg_power = self._profile(m, k, n, dtype)
                    avg_energy = int(t) * avg_power
                    item = (
                        gpu,
                        dtype,
                        m,
                        k,
                        n,
                        int(t),
                        avg_power,
                        avg_energy,
                    )
                    data.append(item)
                    self.data.append(item)
                print(f"Finished profiling m={m}, k={k}")
                # dump data to csv
                df = pd.DataFrame(
                    data,
                    columns=[
                        "gpu",
                        "dtype",
                        "m",
                        "k",
                        "n",
                        "time(us)",
                        "avg_power(W)",
                        "avg_energy(uJ)",
                    ],
                )
                df.to_csv(f"gemm.tmp.{self.worker_id}.csv", index=False, mode='w')
                

        df = pd.DataFrame(
            data,
            columns=[
                "gpu",
                "dtype",
                "m",
                "k",
                "n",
                "time(us)",
                "avg_power(W)",
                "avg_energy(uJ)",
            ],
        )
        return df


def main(gpu: str, num_gpus: int, dtype: str):
    for filename in os.listdir():
        if filename.startswith("power_log_") and filename.endswith(".csv"):
            os.remove(filename)

    profilers = [GemmProfiler.remote(i, num_gpus) for i in range(num_gpus)]
    profiled_data = []
    for i in range(num_gpus):
        profiler = profilers[i]
        data = profiler.profile.remote(gpu, dtype)
        profiled_data.append(data)
    profiled_data = ray.get(profiled_data)
    df = pd.concat(profiled_data, ignore_index=True)
    df = df.sort_values(
        by=["gpu", "dtype", "m", "k", "n"]
    )
    df.to_csv("gemm.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=str, required=True, choices=["V100-PCIE-16GB", "H100-SXM-80GB", "A100-SXM-80GB"]
    )
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "float"])
    args = parser.parse_args()

    print(args)
    main(args.gpu, args.num_gpus, args.dtype)
