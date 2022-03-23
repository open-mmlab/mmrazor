# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import ALGORITHMS
from .spos import SPOS


@ALGORITHMS.register_module()
class GreedyNAS(SPOS):

    def __init__(self, **kwargs):
        super(GreedyNAS, self).__init__(**kwargs)
        self.subnet = self.mutator.sample_subnet()

    def train_step(self, data, optimizer):
        """The iteration step during training.

        In retraining stage, to train subnet like common model. In pre-training
        stage, First to sample a subnet from supernet, then to train the
        subnet.
        """
        if self.retraining:
            outputs = super(GreedyNAS, self).train_step(data, optimizer)
        else:
            self.mutator.set_subnet(self.subnet)
            outputs = super(GreedyNAS, self).train_step(data, optimizer)
        return outputs
