from typing import List

from apex_plus.utils.dtype import DTYPE


class Task:

    def __init__(self) -> None:
        pass

    def get_type(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def get_param_size(tasks: List["Task"], dtype: DTYPE) -> int:
        raise NotImplementedError

    @staticmethod
    def get_kv_token_size(tasks: List["Task"], dtype: DTYPE) -> int:
        raise NotImplementedError

    @classmethod
    def is_attn(cls) -> bool:
        return False
