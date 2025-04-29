from typing import List, Optional

from apex_plus.cluster.device import Device
from apex_plus.cluster.gpu import GPU


class Cluster:
    """Hierarchical cluster."""

    def __init__(
        self,
        level: int,
        children: List["Cluster"] = [],
        devices: List[Device] = [],
        num_nodes: int = 1,
    ) -> None:
        self.level = level
        self.children = children
        self.devices = devices
        self.num_nodes = num_nodes

        if len(self.children) > 0:
            assert len(self.devices) == 0
            num_child_devices = self.children[0].get_num_devices()
            for child in self.children:
                assert child.get_num_devices() == num_child_devices
        else:
            assert self.level == 1
            assert len(self.devices) > 0

    def __repr__(self) -> str:
        return (
            f"Cluster(level={self.level}, children={self.children}, "
            f"devices={self.devices})"
        )

    def get_num_nodes(self) -> int:
        return self.num_nodes

    def get_device(self) -> Device:
        if self.children:
            return self.children[0].get_device()
        else:
            return self.devices[0]

    def get_device_memory_capacity(self) -> int:
        return self.get_device().get_memory_capacity()

    def get_num_devices(self) -> int:
        if len(self.children) > 0:
            return sum(child.get_num_devices() for child in self.children)
        else:
            return len(self.devices)

    def partition(self, n: int) -> Optional[List["Cluster"]]:
        """Partition the cluster into n sub-clusters."""
        if n == 1:
            return [self]

        num_children = len(self.children)
        if num_children == 0:
            # Leaf cluster
            if n > len(self.devices):
                return None
            num_devices_per_partition = len(self.devices) // n
            partitions = []
            for i in range(n):
                devices = self.devices[
                    i * num_devices_per_partition : (i + 1) * num_devices_per_partition
                ]
                partitions.append(Cluster(self.level, devices=devices))
            return partitions
        elif n <= num_children:
            if num_children % n != 0:
                return None
            num_children_per_partition = num_children // n
            if self.num_nodes > 1:
                num_nodes_per_partition = self.num_nodes // n
            else:
                num_nodes_per_partition = 1

            partitions = []
            for i in range(n):
                children = self.children[
                    i
                    * num_children_per_partition : (i + 1)
                    * num_children_per_partition
                ]
                partitions.append(
                    Cluster(
                        self.level,
                        children=children,
                        num_nodes=num_nodes_per_partition,
                    )
                )
            return partitions
        else:
            # n > num_children
            if n % num_children != 0:
                return None
            num_partitions_per_child = n // num_children
            partitions = []
            for child in self.children:
                child_partitions = child.partition(num_partitions_per_child)
                if child_partitions is None:
                    return None
                partitions.extend(child_partitions)
            return partitions

    def is_partitionable(self, n: int) -> bool:
        """Returns if the cluster can be evenly partitioned into n sub-clusters."""
        if n == 1:
            return True
        num_children = len(self.children)
        if num_children == 0:
            # Leaf cluster
            return len(self.devices) % n == 0
        elif n <= num_children:
            return num_children % n == 0
        else:
            # n > num_children
            if n % num_children != 0:
                return False
            num_partitions_per_child = n // num_children
            return self.children[0].is_partitionable(num_partitions_per_child)

    @classmethod
    def from_flat_topology(
        cls,
        gpu_type: str,
        num_devices: int,
    ) -> "Cluster":
        gpus: List[GPU] = []
        for i in range(num_devices):
            gpu = GPU(i, gpu_type)
            gpus.append(gpu)
        return cls(level=1, devices=gpus, num_nodes=1)

    @classmethod
    def from_nvlink_topology(
        cls,
        gpu_type: str,
        num_nodes: int,
        num_devices_per_node: int,
    ) -> "Cluster":
        """Create a cluster with full-mesh NVLink intra-node connections."""
        children: List["Cluster"] = []
        for node_id in range(num_nodes):
            gpus: List[GPU] = []
            for i in range(num_devices_per_node):
                gpu = GPU(node_id * num_devices_per_node + i, gpu_type)
                gpus.append(gpu)
            children.append(cls(level=1, devices=gpus, num_nodes=1))
        if len(children) == 1:
            return children[0]
        else:
            return cls(level=2, children=children, num_nodes=num_nodes)

    @classmethod
    def from_pcie_topology(
        cls,
        gpu_type: str,
        num_nodes: int,
        num_devices_per_node: int,
    ) -> "Cluster":
        """Create a cluster with PCIe intra-node connections."""
        if num_devices_per_node not in [1, 2, 4, 8]:
            raise ValueError(
                "Invalid number of GPUs per node: " f"{num_devices_per_node}"
            )
        children: List["Cluster"] = []
        for node_id in range(num_nodes):
            gpus = [
                GPU(node_id * num_devices_per_node + i, gpu_type)
                for i in range(num_devices_per_node)
            ]

            if len(gpus) == 1 or len(gpus) == 2:
                cluster = cls(level=1, devices=gpus)
            elif len(gpus) == 4:
                cluster = cls(
                    level=2,
                    children=[
                        cls(level=1, devices=gpus[:2]),
                        cls(level=1, devices=gpus[2:]),
                    ],
                )
            elif len(gpus) == 8:
                cluster = cls(
                    level=3,
                    children=[
                        cls(
                            level=2,
                            children=[
                                cls(level=1, devices=gpus[:2]),
                                cls(level=1, devices=gpus[2:4]),
                            ],
                        ),
                        cls(
                            level=2,
                            children=[
                                cls(level=1, devices=gpus[4:6]),
                                cls(level=1, devices=gpus[6:]),
                            ],
                        ),
                    ],
                )
            else:
                assert False
            children.append(cluster)
        if len(children) == 1:
            return children[0]
        else:
            children_level = children[0].level
            return cls(level=children_level + 1, children=children, num_nodes=num_nodes)

    @classmethod
    def from_gpu(
        cls,
        gpu_type: str,
        num_nodes: int,
        num_devices_per_node: int,
    ) -> "Cluster":
        topology = GPU.get_topology(gpu_type)
        if topology == "flat":
            if num_nodes != 1:
                raise ValueError("Flat topology only supports 1 node.")
            return cls.from_flat_topology(
                gpu_type=gpu_type,
                num_devices=num_devices_per_node,
            )
        elif topology == "nvlink":
            return cls.from_nvlink_topology(
                gpu_type=gpu_type,
                num_nodes=num_nodes,
                num_devices_per_node=num_devices_per_node,
            )
        elif topology == "pcie":
            return cls.from_pcie_topology(
                gpu_type=gpu_type,
                num_nodes=num_nodes,
                num_devices_per_node=num_devices_per_node,
            )
        else:
            raise ValueError(f"Invalid topology: {topology}")
