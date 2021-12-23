# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from functools import partial

import mmcv
import yaml
from mmcv.runner import HOOKS, RUNNERS, IterBasedRunner
from mmcv.runner.utils import get_host_info

from ..utils import set_lr


class IterMultiLoader:
    """Multi loaders based on iter."""

    def __init__(self, dataloaders):
        self._dataloaders = dataloaders if isinstance(dataloaders,
                                                      list) else [dataloaders]
        self.iter_loaders = [iter(loader) for loader in self._dataloaders]
        self._epoch = 0

    @property
    def epoch(self):
        """The property of the class."""
        return self._epoch

    @property
    def num_loaders(self):
        """The number of dataloaders."""
        return len(self._dataloaders)

    def __next__(self):
        """Get next iter's data."""
        try:
            data = tuple([next(loader) for loader in self.iter_loaders])
        except StopIteration:
            self._epoch += 1
            for loader in self._dataloaders:
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(self._epoch)
            self.iter_loader = [iter(loader) for loader in self._dataloaders]
            data = tuple([next(loader) for loader in self.iter_loaders])

        return data

    def __len__(self):
        """Get the length of loader."""
        return min([len(loader) for loader in self._dataloaders])


@RUNNERS.register_module()
class MultiLoaderIterBasedRunner(IterBasedRunner):
    """Multi Dataloaders IterBasedRunner.

    There are three differences from IterBasedRunner 1）Support load data from
    multi dataloaders. 2) Support freeze some optimizer's lr update when runner
    has multi optimizers. 3) Add ``search_subnet`` api.
    """

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
            max_iters (int): Specify the max iters.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        # the only difference from IterBasedRunner's ``train``
        iter_loaders = [IterMultiLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

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

        # save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        subnet = self.model.module.search_subnet()
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
