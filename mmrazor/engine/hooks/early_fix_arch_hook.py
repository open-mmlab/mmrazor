# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EarlyArchFixerHook(Hook):
    """Fix the arch params early.
    The EarlyArchFixerHook will fix the value of the max arch param in each
    layer at 1 when the difference between the top-2 arch params is larger
    than the `threshold`.
    NOTE: Only supports differentiable NAS methods at present.

    Args:
        by_epoch (bool): By epoch or by iteration.
            Default: True.
        threshold (float): Threshold to judge whether to fix params or not.
            Default: 0.3 (in paper).
    """

    def __init__(self, by_epoch=True, threshold=0.3, **kwargs):
        self.by_epoch = by_epoch
        self.threshold = threshold

    def before_train_epoch(self, runner):
        """Executed in before_train_epoch stage."""
        if not self.by_epoch:
            return

        model = runner.model.module
        mutator = model.mutator

        if mutator.early_fix_arch:
            if len(mutator.fix_arch_index.keys()) > 0:
                for k, v in mutator.fix_arch_index.items():
                    mutator.arch_params[k].data = v[1]
            for mutable in mutator.mutables:
                arch_param = mutator.arch_params[mutable.key].detach().clone()
                # find the top-2 values of arch_params in the layer
                sort_arch_params = torch.topk(
                    mutator.compute_arch_probs(arch_param), 2)
                argmax_index = (
                    sort_arch_params[0][0] - sort_arch_params[0][1] >=
                    self.threshold)
                # if the max value is large enough, fix current layer.
                if argmax_index:
                    if mutable.key not in mutator.fix_arch_index.keys():
                        mutator.fix_arch_index[mutable.key] = [
                            sort_arch_params[1][0].item(), arch_param]
