# Copyright (c) OpenMMLab. All rights reserved.
import os
import yaml
from typing import Optional, Sequence

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class DMCPSubnetHook(Hook):

    priority = 'VERY_LOW'

    def __init__(self,
                 subnet_sample_num: int = 10,
                 **kwargs) -> None:
        self.subnet_sample_num = subnet_sample_num

    def _save_subnet(self, arch_space_dict, save_path):
        _cfg = dict()
        for k, v in arch_space_dict.items():
            _cfg[k] = int(v)

        with open(save_path, 'w') as file:
            file.write(yaml.dump(_cfg, allow_unicode=True))

    @master_only
    def after_run(self, runner):
        import pdb;pdb.set_trace()
        model = getattr(runner.model, 'module', runner.model)
        runner.logger.info('Sampling...')

        num_sample = self.subnet_sample_num
        root_dir = os.path.join(runner.work_dir, 'model_sample')
        target_flops = model.target_flops

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for i in range(num_sample + 1):
            cur_flops = target_flops * 10
            while cur_flops > target_flops * 1.02 or \
                    cur_flops < target_flops * 0.98:
                model.set_subnet(mode='direct', arch_train=False)
                cur_flops = model.mutator.calc_current_flops(model)

            if i == num_sample:
                model.set_subnet(mode='expected', arch_train=False)
                save_path = os.path.join(root_dir, 'excepted_ch.yaml')
                runner.logger.info(
                    f'Excepted sample(ES) arch with FlOP(MB):{cur_flops}')
            else:
                save_path = os.path.join(root_dir,
                                            'subnet_{}.yaml'.format(i + 1))
                runner.logger.info(
                    f'Driect sample(DS) arch with FlOP(MB): {cur_flops}')
            self._save_subnet(model.mutator.current_choices, save_path)

