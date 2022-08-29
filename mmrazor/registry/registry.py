# Copyright (c) OpenMMLab. All rights reserved.
"""MMRazor provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""
from typing import Any, Optional, Union

from mmengine.config import Config, ConfigDict
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry, build_from_cfg


def build_razor_model_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: 'Registry',
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:

    # TODO relay on mmengine:HAOCHENYE/config_new_feature
    if cfg.get('cfg_path', None) and not cfg.get('type', None):
        from mmengine.hub import get_model
        model = get_model(**cfg)  # type: ignore
        return model

    from mmrazor.structures import load_fix_subnet

    if cfg.get('_fix_subnet_', None):
        fix_subnet = cfg.pop('_fix_subnet_')
        mutable_model = build_from_cfg(cfg, registry, default_args)
        load_fix_subnet(mutable_model, fix_subnet)
        return mutable_model
    else:
        return_architecture = False
        if cfg.get('_return_architecture_', None):
            return_architecture = cfg.pop('_return_architecture_')
        razor_model = build_from_cfg(cfg, registry, default_args)
        if return_architecture:
            return razor_model.architecture
        else:
            return razor_model


# Registries For Runner and the related
# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=MMENGINE_RUNNER_CONSTRUCTORS)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=MMENGINE_LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook', parent=MMENGINE_HOOKS)

# Registries For Data and the related
# manage data-related modules
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)
DATA_SAMPLERS = Registry('data sampler', parent=MMENGINE_DATA_SAMPLERS)
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, build_func=build_razor_model_from_cfg)
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', parent=MMENGINE_MODEL_WRAPPERS)
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', parent=MMENGINE_WEIGHT_INITIALIZERS)

# Registries For Optimizer and the related
# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', parent=MMENGINE_OPTIMIZERS)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optimizer_wrapper', parent=MMENGINE_OPTIM_WRAPPERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS)
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', parent=MMENGINE_PARAM_SCHEDULERS)

# manage all kinds of metrics
METRICS = Registry('metric', parent=MMENGINE_METRICS)

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', parent=MMENGINE_TASK_UTILS)

# Registries For Visualizer and the related
# manage visualizer
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS)
