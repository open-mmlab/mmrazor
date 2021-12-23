# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils import revert_sync_batchnorm

from mmrazor.models.builder import ALGORITHMS
from .spos import SPOS


@ALGORITHMS.register_module()
class DetNAS(SPOS):
    """Implementation of `DetNAS <https://arxiv.org/abs/1903.10979>`_"""

    def __init__(self, **kwargs):
        super(DetNAS, self).__init__(**kwargs)

    def _init_flops(self):
        """Get flops of all modules in supernet in order to easily get each
        subnet's flops."""
        flops_model = copy.deepcopy(self.architecture)
        flops_model = revert_sync_batchnorm(flops_model)
        flops_model.eval()
        flops, params = get_model_complexity_info(flops_model.model.backbone,
                                                  self.input_shape)
        flops_lookup = dict()
        for name, module in flops_model.named_modules():
            flops = getattr(module, '__flops__', 0)
            flops_lookup[name] = flops
        del (flops_model)

        for name, module in self.architecture.named_modules():
            module.__flops__ = flops_lookup[name]
