# Copyright (c) OpenMMLab. All rights reserved.
def set_lr(runner, lr_groups, freeze_optimizers=[]):
    """Set specified learning rate in optimizer."""
    if isinstance(runner.optimizer, dict):
        for k, optim in runner.optimizer.items():
            if k in freeze_optimizers:
                continue
            for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                param_group['lr'] = lr
    else:
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr
