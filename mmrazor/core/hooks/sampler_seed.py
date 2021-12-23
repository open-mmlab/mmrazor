# Copyright (c) Open-MMLab. All rights reserved.
from mmcv.runner import Hook


# @HOOKS.register_module()
class DistSamplerSeedHook(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:``IterBasedRunner`` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def before_epoch(self, runner):
        """Executed in before_epoch stage."""
        if hasattr(runner.data_loader, '_dataloaders'):
            data_loaders = runner.data_loader._dataloaders
        else:
            data_loaders = [runner.data_loader]

        for data_loader in data_loaders:

            if hasattr(data_loader.sampler, 'set_epoch'):
                # in case the data loader uses ``SequentialSampler`` in Pytorch
                data_loader.sampler.set_epoch(runner.epoch)
            elif hasattr(data_loader.batch_sampler.sampler, 'set_epoch'):
                # batch sampler in pytorch warps the sampler as its attributes.
                data_loader.batch_sampler.sampler.set_epoch(runner.epoch)
