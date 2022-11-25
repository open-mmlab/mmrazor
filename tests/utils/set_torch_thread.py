# Copyright (c) OpenMMLab. All rights reserved.
import torch


class SetTorchThread:

    def __init__(self, num_thread: int = -1) -> None:
        self.prev_num_threads = torch.get_num_threads()
        self.num_threads = num_thread

    def __enter__(self):
        if self.num_threads != -1:
            torch.set_num_threads(self.num_threads)

    def __exit__(self, exc_type, exc_value, tb):
        if self.num_threads != -1:
            torch.set_num_threads(self.prev_num_threads)
