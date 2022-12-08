# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import torch
import torch.distributed as dist


class SetDistEnv:

    def __init__(self, using_cuda=False, port=None) -> None:
        self.using_cuda = using_cuda
        if self.using_cuda:
            assert torch.cuda.is_available()
        if port is None:
            port = random.randint(10000, 20000)
        self.port = port

    def __enter__(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.port)

        # initialize the process group
        if self.using_cuda:
            backend = 'nccl'
        else:
            backend = 'gloo'
        dist.init_process_group(backend, rank=0, world_size=1)

    def __exit__(self, exc_type, exc_value, tb):
        dist.destroy_process_group()
