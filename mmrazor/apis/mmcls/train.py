# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
from mmcls.core import DistOptimizerHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner, Fp16OptimizerHook, build_runner
from mmcv.runner.hooks import DistEvalHook, EvalHook

# Differences from mmclassification.
from mmrazor.core.distributed_wrapper import DistributedDataParallelWrapper
from mmrazor.core.hooks import DistSamplerSeedHook
from mmrazor.core.optimizer import build_optimizers
from mmrazor.datasets.utils import split_dataset
from mmrazor.utils import find_latest_checkpoint


def set_random_seed(seed, deterministic=False):
    """Import `set_random_seed` function here was deprecated in v0.3 and will
    be removed in v0.5.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set ``torch.backends.cudnn.deterministic``
            to True and ``torch.backends.cudnn.benchmark`` to False.
            Default: False.
    """
    warnings.warn(
        'Deprecated in v0.3 and will be removed in v0.5, '
        'please import `set_random_seed` directly from `mmrazor.apis`',
        category=DeprecationWarning)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_mmcls_model(model,
                      dataset,
                      cfg,
                      distributed=False,
                      validate=False,
                      timestamp=None,
                      device='cuda',
                      meta=None):
    """Copy from mmclassification and modify some codes.

    This is an ugly implementation, and will be deprecated in the future. In
    the future, there will be only one train api and no longer distinguish
    between mmclassificaiton, mmsegmentation or mmdetection.
    """
    logger = get_root_logger()
    # Difference from mmclassification.
    # Split dataset.
    if cfg.data.get('split', False):
        train_dataset = dataset[0]
        dataset[0] = split_dataset(train_dataset)

    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    # Difference from mmclassification.
    # Build multi dataloaders according the splited datasets.
    data_loaders = list()
    for dset in dataset:
        if isinstance(dset, list):
            data_loader = [
                build_dataloader(item_ds, **train_loader_cfg)
                for item_ds in dset
            ]
        else:
            data_loader = build_dataloader(dset, **train_loader_cfg)

        data_loaders.append(data_loader)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        if cfg.get('use_ddp_wrapper', False):
            # Difference from mmclassification.
            # In some algorithms, the ``optimizer.step()`` is executed in
            # ``train_step``. To rebuilt reducer buckets rightly, there need to
            # use DistributedDataParallelWrapper.
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            # Sets the ``find_unused_parameters`` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cuda':
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        elif device == 'cpu':
            warnings.warn(
                'The argument `device` is deprecated. To use cpu to train, '
                'please refers to https://mmclassification.readthedocs.io/en'
                '/latest/getting_started.html#train-a-model')
            model = model.cpu()
        else:
            raise ValueError(F'unsupported device name {device}.')

    # build optimizers
    # Difference from mmclassification.
    # In some algorithms, there will be multi optimizers.
    optimizer = build_optimizers(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a ``runner`` section, '
            'please set ``runner`` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif (distributed and cfg.optimizer_config is not None
          and 'type' not in cfg.optimizer_config):
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            # Difference from mmclassification.
            # MMRazor's ``DistSamplerSeedHook`` could process multi dataloaders
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_loader_cfg = {
            **loader_cfg,
            'shuffle': False,  # Not shuffle by default
            'sampler_cfg': None,  # Not use sampler by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})

        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # ``EvalHook`` needs to be executed after ``IterTimerHook``.
        # Otherwise, it will cause a bug if use ``IterBasedRunner``.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
