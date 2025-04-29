import enum


class CommType(enum.Enum):
    AllReduce = enum.auto()
    AllGather = enum.auto()
    AllToAll = enum.auto()
    ReduceScatter = enum.auto()


_STR_TO_COMM_TYPE = {
    "AllReduce": CommType.AllReduce,
    "AllGather": CommType.AllGather,
    "AllToAll": CommType.AllToAll,
    "ReduceScatter": CommType.ReduceScatter,
}


def _str_to_comm_type(comm_type: str) -> CommType:
    if comm_type not in _STR_TO_COMM_TYPE:
        raise ValueError(f"Invalid comm_type: {comm_type}")
    return _STR_TO_COMM_TYPE[comm_type]


class CollectiveComm:

    def __init__(
        self,
        comm_type: str,
        num_devices: int,
        size_factor: float = 1.0,
    ) -> None:
        self.comm_type = _str_to_comm_type(comm_type)
        self.num_devices = num_devices
        self.size_factor = size_factor

    def __repr__(self) -> str:
        if self.size_factor == 1:
            return (
                f"CollectiveComm(comm_type={self.comm_type}, "
                f"num_devices={self.num_devices})"
            )
        else:
            return (
                f"CollectiveComm(comm_type={self.comm_type}, "
                f"num_devices={self.num_devices}, "
                f"size_factor={self.size_factor})"
            )
