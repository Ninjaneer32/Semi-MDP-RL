from gym_lifter.envs.fab.wafer import Wafer
from typing import List


class ConveyorBelt:
    def __init__(self, capacity=100):
        self.MAX_CAPACITY = capacity
        self.QUEUE_LEN = 0

        self.QUEUE: List[Wafer] = []   # all elements of the queue must be Wafer objects
        # TODO : data type of a queue : python list -> python deque
        return

    def push(self, wafer: Wafer):
        if self.QUEUE_LEN == self.MAX_CAPACITY:
            # if max capacity is reached, then we cannot let a new wafer in
            return -1
        else:
            self.QUEUE.append(wafer)
            self.QUEUE_LEN += 1
        return 0

    def available(self):
        if self.QUEUE_LEN == self.MAX_CAPACITY:
            return False
        else:
            return True

    def pop(self):
        assert self.QUEUE_LEN > 0
        wafer = self.QUEUE.pop(0)
        self.QUEUE_LEN -= 1
        return wafer



    def reset(self):
        self.QUEUE_LEN = 0
        self.QUEUE = []

    @property
    def cmd_time(self):
        return self.QUEUE[0].cmd_time if self.QUEUE_LEN > 0 else 0.

    @property
    def destination(self) -> int:
        return self.QUEUE[0].destination if self.QUEUE_LEN > 0 else 0

    @property
    def is_empty(self):
        return True if self.QUEUE_LEN == 0 else False

    def __len__(self):
        return self.QUEUE_LEN

    @property
    def is_pod(self):
        return self.QUEUE[0].is_pod if self.QUEUE_LEN > 0 else 0.
