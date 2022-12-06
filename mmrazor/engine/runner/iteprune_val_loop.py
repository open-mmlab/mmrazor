# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmengine import fileio
from mmengine.runner import ValLoop

from mmrazor.registry import LOOPS
from mmrazor.structures import export_fix_subnet


@LOOPS.register_module()
class ItePruneValLoop(ValLoop):
    """Pruning loop for validation. Export fixed subnet configs.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self._save_fix_subnet()
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    def _save_fix_subnet(self):
        """Save model subnet config."""
        fix_subnet = export_fix_subnet(self.model)
        save_name = 'fix_subnet.yaml'
        fileio.dump(fix_subnet, osp.join(self.runner.work_dir, save_name))
        self.runner.logger.info(
            'export finished and '
            f'{save_name} saved in {self.runner.work_dir}.')
