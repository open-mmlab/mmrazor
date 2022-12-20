# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmengine.hooks import Hook

from mmrazor.registry import HOOKS
from mmrazor.utils import print_log


@HOOKS.register_module()
class ChexHook(Hook):

    def before_val(self, runner) -> None:
        config = {}
        for unit in runner.model.mutator.mutable_units:
            config[unit.name] = unit.current_choice
        print_log(json.dumps(config, indent=4))
