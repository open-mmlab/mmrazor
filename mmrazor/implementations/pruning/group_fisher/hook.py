# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner, save_checkpoint
from torch import distributed as torch_dist

from mmrazor.models.algorithms import BaseAlgorithm
from mmrazor.models.mutators.channel_mutator.channel_mutator import \
    ChannelMutator
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.models.task_modules.estimators import ResourceEstimator
from mmrazor.registry import HOOKS, TASK_UTILS
from mmrazor.utils import RuntimeInfo, print_log


def get_model_from_runner(runner):
    """Get the model from a runner."""
    if torch_dist.is_initialized():
        return runner.model.module
    else:
        return runner.model


def is_pruning_algorithm(algorithm):
    """Check whether a model is a pruning algorithm."""
    return isinstance(algorithm, BaseAlgorithm) \
             and isinstance(getattr(algorithm, 'mutator', None), ChannelMutator) # noqa


@HOOKS.register_module()
class PruningStructureHook(Hook):
    """This hook is used to display the structurn information during pruning.

    Args:
        by_epoch (bool, optional): Whether to display structure information
            iteratively by epoch. Defaults to True.
        interval (int, optional): The interval between two structure
            information display.
    """

    def __init__(self, by_epoch=True, interval=1) -> None:

        super().__init__()
        self.by_epoch = by_epoch
        self.interval = interval

    def show_unit_info(self, algorithm):
        """Show unit information of an algorithm."""
        if is_pruning_algorithm(algorithm):
            chices = algorithm.mutator.choice_template
            import json
            print_log(json.dumps(chices, indent=4))

            for unit in algorithm.mutator.mutable_units:
                if hasattr(unit, 'importance'):
                    imp = unit.importance()
                    print_log(
                        f'{unit.name}: \t{imp.min().item()}\t{imp.max().item()}'  # noqa
                    )

    @master_only
    def show(self, runner):
        """Show pruning algorithm information of a runner."""
        algorithm = get_model_from_runner(runner)
        if is_pruning_algorithm(algorithm):
            self.show_unit_info(algorithm)

    # hook points

    def after_train_epoch(self, runner) -> None:
        if self.by_epoch and RuntimeInfo.epoch() % self.interval == 0:
            self.show(runner)

    def after_train_iter(self, runner, batch_idx: int, data_batch,
                         outputs) -> None:
        if not self.by_epoch and RuntimeInfo.iter() % self.interval == 0:
            self.show(runner)


def input_generator_wrapper(model, demp_input: DefaultDemoInput):

    def input_generator(input_shape):
        res = demp_input.get_data(model)
        return res

    return input_generator


@HOOKS.register_module()
class ResourceInfoHook(Hook):
    """This hook is used to display the resource related information and save
    the checkpoint according to a threshold during pruning.

    Args:
        demo_input (dict, optional): the demo input for ResourceEstimator.
            Defaults to DefaultDemoInput([1, 3, 224, 224]).
        interval (int, optional): the interval to check the resource. Defaults
            to 10.
        resource_type (str, optional): the type of resource to check.
            Defaults to 'flops'.
        save_ckpt_thr (list, optional): the threshold to save checkpoint.
            Defaults to [0.5].
        early_stop (bool, optional): whether to stop when all checkpoints have
            been saved according to save_ckpt_thr. Defaults to True.
    """

    def __init__(self,
                 demo_input=DefaultDemoInput([1, 3, 224, 224]),
                 interval=10,
                 resource_type='flops',
                 save_ckpt_thr=[0.5],
                 early_stop=True) -> None:

        super().__init__()
        if isinstance(demo_input, dict):
            demo_input = TASK_UTILS.build(demo_input)

        self.demo_input = demo_input
        self.save_ckpt_thr = sorted(
            save_ckpt_thr, reverse=True)  # big to small
        self.resource_type = resource_type
        self.early_stop = early_stop
        self.estimator: ResourceEstimator = TASK_UTILS.build(
            dict(
                _scope_='mmrazor',
                type='ResourceEstimator',
                flops_params_cfg=dict(
                    input_shape=tuple(demo_input.input_shape), )))
        self.interval = interval
        self.origin_delta = None

    def before_run(self, runner) -> None:
        """Init original_resource."""
        model = get_model_from_runner(runner)
        original_resource = self._evaluate(model)
        print_log(f'get original resource: {original_resource}')

        self.origin_delta = original_resource[self.resource_type]

    # save checkpoint

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        """Check resource after train iteration."""
        if RuntimeInfo.iter() % self.interval == 0 and len(
                self.save_ckpt_thr) > 0:
            model = get_model_from_runner(runner)
            current_delta = self._evaluate(model)[self.resource_type]
            percent = current_delta / self.origin_delta
            if percent < self.save_ckpt_thr[0]:
                self._save_checkpoint(model, runner.work_dir,
                                      self.save_ckpt_thr.pop(0))
        if self.early_stop and len(self.save_ckpt_thr) == 0:
            exit()

    # show info

    @master_only
    def after_train_epoch(self, runner) -> None:
        """Check resource after train epoch."""
        model = get_model_from_runner(runner)
        current_delta = self._evaluate(model)[self.resource_type]
        print_log(
            f'current {self.resource_type}: {current_delta} / {self.origin_delta}'  # noqa
        )

    #

    def _evaluate(self, model: nn.Module):
        """Evaluate the resource required by a model."""
        with torch.no_grad():
            training = model.training
            model.eval()
            res = self.estimator.estimate(
                model,
                flops_params_cfg=dict(
                    input_constructor=input_generator_wrapper(
                        model,
                        self.demo_input,
                    )))
            if training:
                model.train()
            return res

    @master_only
    def _save_checkpoint(self, model, path, delta_percent):
        """Save the checkpoint  of a model."""
        ckpt = {'state_dict': model.state_dict()}
        save_path = f'{path}/{self.resource_type}_{delta_percent:.2f}.pth'
        save_checkpoint(ckpt, save_path)
        print_log(
            f'Save checkpoint to {save_path} with {self._evaluate(model)}'  # noqa
        )
