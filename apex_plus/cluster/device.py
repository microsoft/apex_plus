class Device:

    def __init__(
        self,
        device_id: int,
        device_type: str,
    ) -> None:
        raise NotImplementedError

    def get_memory_capacity(self) -> int:
        raise NotImplementedError
