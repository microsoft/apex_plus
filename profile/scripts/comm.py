import argparse
import time
import os

import cupy as cp
from cupy.cuda import nccl
import numpy as np
import ray

# import pandas as pd
import csv
from typing import List, Tuple

from pynvml import *

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30

gpu = "A100-SXM-80GB"
data_type = cp.float16

_NCCL_DT = {  # convert data type to NCCL format to run experiments
    cp.float16: nccl.NCCL_FLOAT16,
    cp.float32: nccl.NCCL_FLOAT32,
}


def do_all_reduce(comm, in_buffer, out_buffer, dtype):
    comm.allReduce(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        _NCCL_DT[dtype],
        0,
        cp.cuda.Stream.null.ptr,
    )


def do_all_gather(comm, in_buffer, out_buffer, dtype):
    comm.allGather(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        _NCCL_DT[dtype],
        cp.cuda.Stream.null.ptr,
    )


def do_all_to_all(comm, in_buffer, out_buffer, dtype):
    world_size = comm.size()
    nccl.groupStart()
    for i in range(comm.size()):
        comm.send(
            in_buffer.data.ptr,
            in_buffer.size // world_size,
            _NCCL_DT[dtype],
            i,
            cp.cuda.Stream.null.ptr,
        )
        comm.recv(
            out_buffer.data.ptr,
            out_buffer.size // world_size,
            _NCCL_DT[dtype],
            i,
            cp.cuda.Stream.null.ptr,
        )
    nccl.groupEnd()


def do_reduce_scatter(comm, in_buffer, out_buffer, dtype):
    comm.reduceScatter(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        out_buffer.size,
        _NCCL_DT[dtype],
        nccl.NCCL_SUM,
        cp.cuda.Stream.null.ptr,
    )


def do_send_recv(comm, buf, is_sender, dtype):
    if is_sender:
        comm.send(buf.data.ptr, buf.size, _NCCL_DT[dtype], 1, cp.cuda.Stream.null.ptr)
    else:
        comm.recv(buf.data.ptr, buf.size, _NCCL_DT[dtype], 0, cp.cuda.Stream.null.ptr)


@ray.remote(num_gpus=1)
class GpuHost:
    def __init__(self, rank, world_size, nccl_uuid_list):
        self.rank = rank
        self.world_size = world_size
        self.nccl_uuid_list = nccl_uuid_list
        self.ct = 0

        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)

    def init_communicator(self, groups):
        if np.max(groups) >= self.world_size:
            return None
        if len(set(np.ravel(groups))) < len(np.ravel(groups)):
            return None

        comm = None
        for group in groups:
            nccl_uuid = self.nccl_uuid_list[self.ct]
            self.ct += 1
            for device_id in group:
                if self.rank == device_id:
                    assert comm is None
                    comm = cp.cuda.nccl.NcclCommunicator(
                        len(group), nccl_uuid, group.index(self.rank)
                    )

        cp.cuda.Device(0).synchronize()
        return comm

    def profile_allreduce(self, size, dtype, groups, gpu_freq, file_path):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size), dtype)
        out_buffer = cp.ones(int(size), dtype)

        nccl.NCCL_FLOAT16

        do_all_reduce(comm, in_buffer, out_buffer, dtype)
        do_all_reduce(comm, in_buffer, out_buffer, dtype)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_reduce(comm, in_buffer, out_buffer, dtype)
        cp.cuda.Device(0).synchronize()
        toc = time.time()
        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = 2 * array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost

            data_dtype = "float" if dtype == cp.float32 else "half"
            data = [
                gpu,
                len(groups),
                num_devices,
                data_dtype,
                int(array_size / KB),
                int(time_cost * 1.0e6),
                gpu_freq,
            ]
            with open(file_path, "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(data)

            print(
                f"AllReduce: {groups}\tBytes: {array_size / KB} KB\t"
                f"Time: {time_cost:.6f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
            )

    def profile_reduce_scatter(self, size, dtype, groups, file_path):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size), dtype)
        out_buffer = cp.ones(int(size) // len(groups[0]), dtype)

        do_reduce_scatter(comm, in_buffer, out_buffer, dtype)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_reduce_scatter(comm, in_buffer, out_buffer, dtype)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            data_dtype = "float" if dtype == cp.float32 else "half"
            data = [
                gpu,
                len(groups),
                num_devices,
                data_dtype,
                int(array_size / KB),
                int(time_cost * 1.0e6),
            ]
            with open(file_path, "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            print(
                f"ReduceScatter: {groups}\tBytes: {array_size / KB} KB\t"
                f"Time: {time_cost:.6f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
            )

    def profile_allgather(self, size, dtype, groups, file_path):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size) // len(groups[0]), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_gather(comm, in_buffer, out_buffer, dtype)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_gather(comm, in_buffer, out_buffer, dtype)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            data_dtype = "float" if dtype == cp.float32 else "half"
            data = [
                gpu,
                len(groups),
                num_devices,
                data_dtype,
                int(array_size / KB),
                int(time_cost * 1.0e6),
            ]
            with open(file_path, "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            print(
                f"AllGather: {groups}\tBytes: {array_size / KB} KB\t"
                f"Time: {time_cost:.6f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
            )

    def profile_alltoall(self, size, dtype, groups, file_path):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_to_all(comm, in_buffer, out_buffer, dtype)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_to_all(comm, in_buffer, out_buffer, dtype)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            data_dtype = "float" if dtype == cp.float32 else "half"
            data = [
                gpu,
                len(groups),
                num_devices,
                data_dtype,
                int(array_size / KB),
                int(time_cost * 1.0e6),
            ]
            with open(file_path, "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            print(
                f"AllToAll: {groups}\tBytes: {array_size / KB} KB\t"
                f"Time: {time_cost:.6f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
            )

    def profile_send_recv(self, size, dtype, from_rank, to_rank, file_path):
        groups = [[from_rank, to_rank]]
        comm = self.init_communicator(groups)
        if comm is None:
            return

        buf = cp.ones(int(size), dtype)
        do_send_recv(comm, buf, self.rank == from_rank, dtype)
        do_send_recv(comm, buf, self.rank == from_rank, dtype)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_send_recv(comm, buf, self.rank == from_rank, dtype)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == from_rank:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            data_dtype = "float" if dtype == cp.float32 else "half"
            data = [
                gpu,
                len(groups),
                len(groups[0]),
                data_dtype,
                int(array_size / KB),
                int(time_cost * 1.0e6),
            ]
            with open(file_path, "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            print(
                f"SendRecv: {groups}\tBytes: {array_size / KB} KB\t"
                f"Time: {time_cost:.6f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
            )

    def profile_multi_send_recv(self, size, dtype, groups, file_path):
        comm = self.init_communicator(groups)
        time.sleep(1)
        comm_sync = self.init_communicator([list(np.ravel(groups))])
        if comm is None or comm_sync is None:
            return

        assert all(len(group) == 2 for group in groups)

        senders = set(group[0] for group in groups)
        receivers = set(group[1] for group in groups)

        buf = cp.ones(int(size), dtype)
        buf_sync = cp.ones(1, dtype)

        do_send_recv(comm, buf, self.rank in senders, dtype)
        do_send_recv(comm, buf, self.rank in senders, dtype)
        do_all_reduce(comm_sync, buf_sync, buf_sync, dtype)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_send_recv(comm, buf, self.rank in senders, dtype)
        do_all_reduce(comm_sync, buf_sync, buf_sync, dtype)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == groups[0][0]:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = len(groups) * communication_size / time_cost
            data = [
                gpu,
                len(groups),
                len(groups[0]),
                dtype,
                int(array_size / KB),
                int(time_cost * 1.0e6),
            ]
            with open(file_path, "a", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(data)
            print(
                f"SendRecv: {groups}\tBytes: {array_size / KB} KB\t"
                f"Time: {time_cost:.6f} s\tBandwidth: {bandwidth / GB:.2f} GB/s"
            )

    def profile(self):
        data_type = cp.float16
        header = [
            "gpu",
            "num_nodes",
            "num_gpus_per_node",
            "dtype",
            "size(kb)",
            "time(us)",
            "gpu_freq",
            "energy(J)",
        ]
        gpu_frequencies = [800, 1200, 1600, 1980]

        # # All-reduce
        # file_path = "all_reduce.csv"
        # with open(file_path, "w", encoding="UTF8") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)
        # # for freq in gpu_frequencies:
        # #     nvmlDeviceSetGpuLockedClocks(self.handle, freq, freq)
        # #     time.sleep(2)
        # freq = 0
        # for i in range(8, 27):
        #     for k in [1, 2, 4]:
        #         self.profile_allreduce(
        #             1 << i,
        #             data_type,
        #             [list(range(0, self.world_size, k))],
        #             freq,
        #             file_path,
        #         )

        # # All-gather
        # file_path = "all_gather.csv"
        # with open(file_path, 'w', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)
        # for i in range(8, 27):
        #     for k in [1, 2, 4]:
        #         self.profile_allgather(1 << i, data_type, [list(range(0, self.world_size, k))], file_path)

        # # All-to-all
        # file_path = "alltoall.csv"
        # with open(file_path, 'w', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)
        # for i in range(8, 27):
        #     for k in [1, 2, 4]:
        #         self.profile_alltoall(1 << i, data_type, [list(range(0, self.world_size, k))], file_path)

        # # Reduce-Scatter
        # file_path = "reduce_scatter.csv"
        # with open(file_path, 'w', encoding='UTF8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)
        # for i in range(8, 27):
        #     for k in [1, 2, 4]:
        #         self.profile_reduce_scatter(1 << i, data_type, [list(range(0, self.world_size, k))], file_path)

        # Single Send-recv
        file_path = "sendrecv.csv"
        with open(file_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        for i in range(8, 27):
            self.profile_send_recv(1 << i, data_type, 0, self.world_size - 1,file_path)

        # multiple p2p Send-recv
        # for i in range(29, 30):
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, 1], [2, 3]])
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 4], [1, self.world_size - 3]])
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 2], [1, self.world_size - 1]])
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 4], [1, self.world_size - 3], [2, self.world_size - 2], [3, self.world_size - 1]])
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 8], [1, self.world_size - 7], [2, self.world_size - 6], [3, self.world_size - 5]])
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 8], [1, self.world_size - 7], [2, self.world_size - 6], [3, self.world_size - 5],
        #                                                       [4, self.world_size - 4], [5, self.world_size - 3], [6, self.world_size - 2], [7, self.world_size - 1]])

        # for handle in handles:
        #     nvmlDeviceResetGpuLockedClocks(handle)

        nvmlShutdown()

    def sync(self):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--efa", action="store_true", help="Use AWS EFS on p3.24 or p4.24 instances"
    )
    parser.add_argument(
        "--ib", action="store_true", help="Use InfiniBand for NCCL communcation"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print nccl debug information"
    )
    parser.add_argument(
        "--gpu", type=str, required=True, choices=["V100-PCIE-16GB", "H100-SXM-80GB", "A100-SXM-80GB"]
    )
    parser.add_argument("--dtype", type=str, default="half", choices=["half", "float"])
    args = parser.parse_args()

    gpu = args.gpu
    data_type = cp.float32 if args.dtype == "float" else cp.half

    # ray.init(address="auto")
    ray.init()
    num_gpus = int(ray.cluster_resources()["GPU"])

    nccl_uuid_list = [cp.cuda.nccl.get_unique_id() for _ in range(500)]

    workers = []
    for i in range(num_gpus):
        if args.efa:
            env_vars = {
                "FI_PROVIDER": "efa",
                "FI_EFA_USE_DEVICE_RDMA": "1",
                "LD_LIBRARY_PATH": os.environ.get(
                    "LD_LIBRARY_PATH", ""
                ),  # For libnccl-net.so
                "NCCL_PROTO": "simple",
            }
        elif args.ib:
            env_vars = {
                "NCCL_SOCKET_NTHREADS": "4",
                "NCCL_NSOCKS_PERTHREAD": "4",
                "NCCL_IB_HCA": "mlx5,ibp",  # Change this to align with your IB interface name
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }
        else:
            env_vars = {
                "NCCL_SOCKET_NTHREADS": "4",
                "NCCL_NSOCKS_PERTHREAD": "4",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }

        if args.debug:
            env_vars["NCCL_DEBUG"] = "INFO"

        workers.append(
            GpuHost.options(runtime_env={"env_vars": env_vars}).remote(
                i, num_gpus, nccl_uuid_list
            )
        )

    ray.get([w.profile.remote() for w in workers])
    ray.get([w.sync.remote() for w in workers])
