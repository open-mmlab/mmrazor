# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
from functools import partial

import mmcv
import yaml
from mmcv.runner import HOOKS, RUNNERS, EpochBasedRunner

from ..utils import set_lr


class EpochMultiLoader:
    """Multi loaders based on epoch."""

    def __init__(self, dataloaders):
        self._dataloaders = dataloaders
        self.iter_loaders = [iter(loader) for loader in self._dataloaders]

    @property
    def num_loaders(self):
        """The number of dataloaders."""
        return len(self._dataloaders)

    def __iter__(self):
        """Return self when executing __iter__."""
        return self

    def __next__(self):
        """Get next iter's data."""
        data = tuple([next(loader) for loader in self.iter_loaders])

        return data

    def __len__(self):
        """Get the length of loader."""
        return min([len(loader) for loader in self._dataloaders])


@RUNNERS.register_module()
class MultiLoaderEpochBasedRunner(EpochBasedRunner):
    """Multi Dataloaders EpochBasedRunner.

    There are three differences from EpochBaseRunner： 1）Support load data from
    multi dataloaders. 2) Support freeze some optimizer's lr update when runner
    has multi optimizers. 3) Add ``search_subnet`` api.
    """

    def train(self, data_loader, **kwargs):
        """Rewrite the ``train`` of ``EpochBasedRunner``."""
        self.model.train()
        self.mode = 'train'

        # the only difference from EpochBasedRunner's ``train``
        if isinstance(data_loader, list):
            self.data_loader = EpochMultiLoader(data_loader)
        else:
            self.data_loader = data_loader

        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')

        self._epoch += 1

    def register_lr_hook(self, lr_config):
        """Resister a hook for setting learning rate.

        Args:
            lr_config (dict): Config for setting learning rate.
        """
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            if 'freeze_optimizers' in lr_config:
                freeze_optimizers = lr_config.pop('freeze_optimizers')
            else:
                freeze_optimizers = []
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for ``CosineAnnealingLrUpdater``,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = mmcv.build_from_cfg(lr_config, HOOKS)

            # modify the hook's ``_set_lr``
            # the only difference from BasedRunner's ``register_lr_hook``
            hook._set_lr = partial(set_lr, freeze_optimizers=freeze_optimizers)
        else:
            hook = lr_config

        self.register_hook(hook, priority=10)

    def search_subnet(self,
                      out_dir,
                      filename_tmpl='epoch_{}.yaml',
                      create_symlink=True):
        """Search the best subnet.

        Args:
            out_dir (str): The directory that subnets are saved.
            filename_tmpl (str, optional): The subnet filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.yaml'.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.yaml" to point to the latest subnet.
                Defaults to True.
        """

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)

        algorithm = self.model.module if hasattr(self.model,
                                                 'module') else self.model
        mutator = algorithm.mutator.module if hasattr(
            algorithm.mutator, 'module') else algorithm.mutator
        subnet = mutator.search_subnet()
        with open(filepath, 'w') as f:
            yaml.dump(subnet, f)
        # in some environments, ``os.symlink`` is not supported, you may need
        # to set ``create_symlink`` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.yaml')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
