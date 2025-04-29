from typing import List

from apex_plus.parallel.comm import CollectiveComm, CommType
from apex_plus.parallel.schedule import CellSchedule


def is_reshardable(
    c1: CellSchedule,
    c2: CellSchedule,
) -> bool:
    """Returns whether resharding is possible."""
    # NOTE(woosuk): Currently, we only support resharding when the number of
    # replicas is a multiple of the number of replicas in the previous cell.
    return (
        c1.num_replicas % c2.num_replicas == 0 or c2.num_replicas % c1.num_replicas == 0
    )


def get_reshard_comm(
    c1: CellSchedule,
    c2: CellSchedule,
) -> List[CollectiveComm]:
    """Returns collective communications for resharding.

    Args:
        c1: A cell schedule that precedes c2.
        c2: A cell schedule that follows c1.

    Returns:
        Collective communications for resharding.
    """
    assert is_reshardable(c1, c2)
    c1_comm = c1.task_mapping.collective_comm
    n = c1_comm.num_devices
    # Handle MoE with template 0.
    if _is_moe_template0(c1):
        assert c2.cell.get_name() != "MoE"
        topk = c1.cell.topk
        capacity_factor = c1.cell.capacity_factor
        size_factor = topk * capacity_factor

        if c1.num_replicas % c2.num_replicas == 0:
            k = c1.num_replicas // c2.num_replicas
            # AllGather(n) -> AlltoAll(n) + AllGather(n * k)
            size_factor = min(size_factor, n)
            return [
                CollectiveComm(
                    num_devices=n, comm_type="AllToAll", size_factor=size_factor
                ),
                CollectiveComm(
                    num_devices=n * k,
                    comm_type="AllGather",
                    size_factor=1 / size_factor,
                ),
            ]
        elif c2.num_replicas % c1.num_replicas == 0:
            k = c2.num_replicas // c1.num_replicas
            # AllGather(n) -> AlltoAll(n) + AllGather(n / k)
            size_factor = min(size_factor, n)
            return [
                CollectiveComm(
                    num_devices=n, comm_type="AllToAll", size_factor=size_factor
                ),
                CollectiveComm(
                    num_devices=n // k,
                    comm_type="AllGather",
                    size_factor=1 / size_factor,
                ),
            ]
        else:
            assert False
    elif _is_moe_template0(c2):
        topk = c2.cell.topk
        capacity_factor = c2.cell.capacity_factor
        size_factor = topk * capacity_factor

        if c1.num_replicas % c2.num_replicas == 0:
            k = c1.num_replicas // c2.num_replicas
            if c1_comm.comm_type == CommType.AllReduce:
                # AllReduce(n) -> ReduceScatter(n) + AllToAll(n * k)
                size_factor = min(size_factor, n * k)
                return [
                    CollectiveComm(num_devices=n, comm_type="ReduceScatter"),
                    CollectiveComm(
                        num_devices=n * k, comm_type="AllToAll", size_factor=size_factor
                    ),
                ]
            else:
                raise NotImplementedError()
        elif c2.num_replicas % c1.num_replicas == 0:
            k = c2.num_replicas // c1.num_replicas
            if c1_comm.comm_type == CommType.AllReduce:
                # AllReduce(n) -> ReduceScatter(n) + AllToAll(n / k)
                size_factor = min(size_factor, n // k)
                return [
                    CollectiveComm(num_devices=n, comm_type="ReduceScatter"),
                    CollectiveComm(
                        num_devices=n // k,
                        comm_type="AllToAll",
                        size_factor=size_factor,
                    ),
                ]
            else:
                raise NotImplementedError()
        else:
            assert False

    if c1.cell.get_name() == "SwiGLU":
        # NOTE we only need to handle the cases of going from SwiGLU to MHA;
        # for MHA to SwiGLU, no new cases needs to be handled
        if c1.num_replicas == c2.num_replicas:
            # AllReduce(n): between 2nd and 3rd linear layer
            # AllGather(n): between MLP and MHA
            return [
                CollectiveComm(num_devices=n, comm_type="AllReduce"),
                CollectiveComm(num_devices=n, comm_type="AllGather"),
            ]
        elif c2.num_replicas % c1.num_replicas == 0:
            k = c2.num_replicas // c1.num_replicas
            return [
                CollectiveComm(num_devices=n, comm_type="AllReduce"),
                CollectiveComm(num_devices=n, comm_type="AllToAll"),
                CollectiveComm(num_devices=n // k, comm_type="AllGather"),
            ]
        elif c1.num_replicas % c2.num_replicas == 0:
            k = c1.num_replicas // c2.num_replicas
            return [
                CollectiveComm(num_devices=n, comm_type="AllReduce"),
                CollectiveComm(num_devices=n * k, comm_type="AllGather"),
            ]

    if c1.num_replicas == c2.num_replicas:
        return [c1.task_mapping.collective_comm]
    elif c2.num_replicas % c1.num_replicas == 0:
        k = c2.num_replicas // c1.num_replicas
        if c1_comm.comm_type == CommType.AllReduce:
            # AllReduce(n) -> ReduceScatter(n) + AllGather(n / k)
            return [
                CollectiveComm(num_devices=n, comm_type="ReduceScatter"),
                CollectiveComm(num_devices=n // k, comm_type="AllGather"),
            ]
        else:
            raise NotImplementedError()
    elif c1.num_replicas % c2.num_replicas == 0:
        k = c1.num_replicas // c2.num_replicas
        if c1_comm.comm_type == CommType.AllReduce:
            # AllReduce(n) -> ReduceScatter(n) + AllGather(n * k)
            return [
                CollectiveComm(num_devices=n, comm_type="ReduceScatter"),
                CollectiveComm(num_devices=n * k, comm_type="AllGather"),
            ]
        else:
            print(c1_comm.comm_type)
            raise NotImplementedError()
    else:
        assert False


def _is_moe_template0(cell_schedule: CellSchedule) -> bool:
    if (
        cell_schedule.cell.get_name() != "MoE"
        and cell_schedule.cell.get_name() != "SwiMoE"
    ):
        return False
    comm = cell_schedule.task_mapping.collective_comm
    return comm.comm_type == CommType.AllGather
