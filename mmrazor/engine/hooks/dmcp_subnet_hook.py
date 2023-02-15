# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from mmrazor.structures import export_fix_subnet


@HOOKS.register_module()
class DMCPSubnetHook(Hook):
    """Dump subnet periodically.

    Args:
        subnet_sample_num (int):The number of networks sampled,
            the last of which is the sub-network sampled in ``expected``
            mode and the others are sampled in ``direct`` mode.
            Defaults to 10.
    """

    priority = 'VERY_LOW'

    def __init__(self, subnet_sample_num: int = 10, **kwargs) -> None:
        self.subnet_sample_num = subnet_sample_num

    def _save_subnet(self, model, runner, save_path):
        """Save the sampled sub-network config."""
        fix_subnet, _ = export_fix_subnet(
            model,
            export_subnet_mode='mutator',
            slice_weight=True,
        )
        fix_subnet = json.dumps(fix_subnet, indent=4, separators=(',', ':'))
        with open(save_path, 'w') as file:
            file.write(fix_subnet)

        runner.logger.info('export finished and '
                           f'{save_path} saved in {runner.work_dir}.')

    def after_run(self, runner):
        """Save the sampled subnet under target FLOPs.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = getattr(runner.model, 'module', runner.model)
        runner.logger.info('Sampling...')

        num_sample = self.subnet_sample_num
        root_dir = os.path.join(runner.work_dir, 'model_sample')
        target_flops = model.target_flops * 1e6

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for i in range(num_sample + 1):
            cur_flops = target_flops * 10
            while cur_flops > target_flops * 1.05 or \
                    cur_flops < target_flops * 0.95:
                model.set_subnet(mode='direct', arch_train=False)
                cur_flops = model.calc_current_flops()

            if i == num_sample:
                model.set_subnet(mode='expected', arch_train=False)
                save_path = os.path.join(root_dir, 'excepted_ch.json')
                runner.logger.info(
                    f'Excepted sample(ES) arch with FlOP(MB):{cur_flops}')
            else:
                save_path = os.path.join(root_dir,
                                         'subnet_{}.json'.format(i + 1))
                runner.logger.info(
                    f'Driect sample(DS) arch with FlOP(MB): {cur_flops/1e6}')
            self._save_subnet(model, runner, save_path)
