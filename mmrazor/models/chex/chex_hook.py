# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmengine import dist
from mmengine.hooks import Hook

from mmrazor.registry import HOOKS
from mmrazor.utils import print_log


@HOOKS.register_module()
class ChexHook(Hook):

    @classmethod
    def algorithm(cls, runner):
        if dist.is_distributed():
            return runner.model.module
        else:
            return runner.model

    def before_val(self, runner) -> None:
        algorithm = self.algorithm(runner)
        if dist.get_rank() == 0:
            config = {}
            for unit in algorithm.mutator.mutable_units:
                config[unit.name] = unit.current_choice
            print_log(json.dumps(config, indent=4))
            print_log(f'growth_ratio: {algorithm.growth_ratio}')
