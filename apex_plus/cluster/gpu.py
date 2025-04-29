import enum

from apex_plus.cluster.device import Device
from apex_plus.utils.dtype import DTYPE

_GB = 1 << 30


class GPUType(enum.Enum):
    # A10 = enum.auto()
    # A100_40GB = enum.auto()
    # A100_80GB = enum.auto()
    V100_PCIE_16GB = enum.auto()
    H100_SXM_80GB = enum.auto()
    H200_SXM_141GB = enum.auto()


_GPU_REGISTRY = {
    # "A10": GPUType.A10,
    # "A100-40GB": GPUType.A100_40GB,
    # "A100-80GB": GPUType.A100_80GB,
    "V100-PCIE-16GB": GPUType.V100_PCIE_16GB,
    "H100-SXM-80GB": GPUType.H100_SXM_80GB,
    "H200-SXM-141GB": GPUType.H200_SXM_141GB,
}

_GPU_TYPE_TO_MEMORY_GB = {
    # GPUType.A10: 24 * _GB,
    # GPUType.A100_40GB: 40 * _GB,
    # GPUType.A100_80GB: 80 * _GB,
    GPUType.V100_PCIE_16GB: 16 * _GB,
    GPUType.H100_SXM_80GB: 80 * _GB,
    GPUType.H200_SXM_141GB: 141 * _GB,
}

_GPU_TYPE_TO_TOPOLOGY = {
    # GPUType.A10: "pcie",
    # GPUType.A100_40GB: "nvlink",
    # GPUType.A100_80GB: "nvlink",
    GPUType.V100_PCIE_16GB: "pcie",
    GPUType.H100_SXM_80GB: "nvlink",
    GPUType.H200_SXM_141GB: "nvlink",
}

_GPU_PEAK_FLOPS = {
    GPUType.V100_PCIE_16GB: {
        DTYPE.FLOAT32: 7e12,
        DTYPE.FLOAT16: 14e12,
    },
    GPUType.H100_SXM_80GB: {
        DTYPE.FLOAT32: 67e12,
        DTYPE.FLOAT16: 1979e12,
        DTYPE.FLOAT8: 3958e12,
    },
    GPUType.H200_SXM_141GB: {
        DTYPE.FLOAT32: 67e12,
        DTYPE.FLOAT16: 1979e12,
        DTYPE.FLOAT8: 3958e12,
    },
}

_GPU_PEAK_MEM_BANDWIDTH = {
    GPUType.V100_PCIE_16GB: 900e9,
    GPUType.H100_SXM_80GB: 3.35e12,
    GPUType.H200_SXM_141GB: 4.8e12,
}


class GPU(Device):

    def __init__(
        self,
        device_id: int,
        device_type: str,
    ) -> None:
        self.device_id = device_id
        device_type = device_type.upper()
        self.device_type = device_type

        if device_type not in _GPU_REGISTRY:
            raise ValueError(f"Unknown GPU: {device_type}")
        self.gpu_type = _GPU_REGISTRY[device_type]
        self.total_memory = _GPU_TYPE_TO_MEMORY_GB[self.gpu_type]
        self.peak_flops = _GPU_PEAK_FLOPS[self.gpu_type]
        self.peak_mem_bandwidth = _GPU_PEAK_MEM_BANDWIDTH[self.gpu_type]

    def get_memory_capacity(self) -> int:
        return self.total_memory

    def __repr__(self) -> str:
        return (
            f"GPU(device_id={self.device_id}, "
            f"device_type={self.device_type},"
            f"peak_flops={self.peak_flops},"
            f"peak_mem_bandwidth={self.peak_mem_bandwidth},"
            f"device_type={self.device_type},"
        )

    @staticmethod
    def get_topology(gpu: str) -> str:
        gpu_type = _GPU_REGISTRY[gpu.upper()]
        return _GPU_TYPE_TO_TOPOLOGY[gpu_type]
