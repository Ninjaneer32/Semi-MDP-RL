from typing import Optional, Tuple
from gym_lifter.envs.fab.wafer import Wafer
from copy import deepcopy

class Rack:
    def __init__(self):
        self.upper_fork: Optional[Wafer] = None
        self.lower_fork: Optional[Wafer] = None

        self.is_upper_loaded: bool = False
        self.is_lower_loaded: bool = False

    def release_lower_fork(self) -> Wafer:
        assert self.is_lower_loaded
        wafer_released = deepcopy(self.lower_fork)
        self.lower_fork = None
        self.is_lower_loaded = False
        return wafer_released

    def release_upper_fork(self) -> Wafer:
        assert self.is_upper_loaded
        wafer_released = deepcopy(self.upper_fork)
        self.upper_fork = None
        self.is_upper_loaded = False
        return wafer_released

    def load_lower(self, wafer: Wafer):
        assert not self.is_lower_loaded
        self.lower_fork = wafer
        self.is_lower_loaded = True
        return

    def load_upper(self, wafer: Wafer):
        assert not self.is_upper_loaded
        self.upper_fork = wafer
        self.is_upper_loaded = True
        return

    def load_pod(self, wafer: Wafer):
        # rule : POD is always taken by the lower fork
        assert wafer.is_pod and not (self.is_upper_loaded or self.is_lower_loaded)
        self.lower_fork = wafer
        self.is_lower_loaded = True

    def reset(self):
        self.upper_fork = None
        self.lower_fork = None
        self.is_lower_loaded = False
        self.is_upper_loaded = False

    @property
    def destination(self) -> Tuple[int, int]:
        destination1 = self.lower_fork.destination if self.is_lower_loaded else 0
        destination2 = self.upper_fork.destination if self.is_upper_loaded else 0
        return destination1, destination2


    @property
    def is_pod_loaded(self) -> bool:
        if self.is_lower_loaded:
            flag = self.lower_fork.is_pod
        elif self.is_upper_loaded:
            flag = self.upper_fork.is_pod
        else:
            flag = False
        return flag
