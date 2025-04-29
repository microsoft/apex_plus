from apex_plus.cluster.device import Device

_LOGICAL_DEVICE_TYPE = "LOGICAL"


class LogicalDevice(Device):

    def __init__(
        self,
        device_id: int,
        device_type: str,
    ) -> None:
        self.device_id = device_id
        if device_type.upper() != _LOGICAL_DEVICE_TYPE:
            raise ValueError(f"Unknown device type: {device_type}")
        self.device_type = _LOGICAL_DEVICE_TYPE
