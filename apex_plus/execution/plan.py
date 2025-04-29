from typing import List

from apex_plus.cluster.cluster import Cluster
from apex_plus.parallel.schedule import ParallelSchedule


class ExecutionPlan:

    def __init__(
        self,
        parallel_schedule: ParallelSchedule,
        stage_clusters: List[Cluster],  # stage -> cluster
        cell_clusters: List[List[Cluster]],  # [cell][cell replica] -> cluster
    ) -> None:
        self.parallel_schedule = parallel_schedule
        self.stage_clusters = stage_clusters
        self.cell_clusters = cell_clusters

    def __repr__(self) -> str:
        return (
            f"ExecutionPlan(parallel_schedule={self.parallel_schedule}, "
            f"stage_clusters={self.stage_clusters}, "
            f"cell_clusters={self.cell_clusters})"
        )
