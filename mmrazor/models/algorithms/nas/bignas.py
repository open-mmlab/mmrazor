# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.utils import ValidFixMutable
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[BaseMutator, Dict]
VALID_MUTATORS_TYPE = Dict[str, Union[BaseMutator, Dict]]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class BigNAS(BaseAlgorithm):

    strategy_groups: Dict[str, List] = {
        'sandwich2': ['max', 'min'],
        'sandwich3': ['max', 'random', 'min'],
        'sandwich4': ['max', 'random', 'random1', 'min'],
        'batch-sandwich4': [['max', 'random', 'random', 'min']],
        'batch-sandwich4_distill': [['max'],
                                    ['max', 'random', 'random', 'min']]
    }

    def __init__(self,
                 mutators: VALID_MUTATORS_TYPE,
                 distiller: VALID_DISTILLER_TYPE,
                 architecture: Union[BaseModel, Dict],
                 fix_subnet: Optional[ValidFixMutable] = None,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 strategy: str = 'sandwich4',
                 init_cfg: Optional[Dict] = None,
                 num_samples: int = 2,
                 drop_prob: float = 0.2) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        if isinstance(mutators, dict):
            built_mutators: Dict = dict()
            for name, mutator_cfg in mutators.items():
                if 'parse_cfg' in mutator_cfg and isinstance(
                        mutator_cfg['parse_cfg'], dict):
                    assert mutator_cfg['parse_cfg'][
                        'type'] == 'Predefined', \
                            'autoformer only support predefined.'
                mutator: BaseMutator = MODELS.build(mutator_cfg)
                built_mutators[name] = mutator
                mutator.prepare_from_supernet(self.architecture)
            self.mutators = built_mutators
        else:
            raise TypeError('mutator should be a `dict` but got '
                            f'{type(mutator)}')

        self.strategy = strategy
        self.distiller = self._build_distiller(distiller)
        self.distiller.prepare_from_teacher(self.architecture)
        self.distiller.prepare_from_student(self.architecture)

        self.num_samples = num_samples
        self.drop_prob = drop_prob
        self.is_supernet = True
        self._optim_wrapper_count_status_reinitialized = False

        # BigNAS supports supernet training and subnet retraining.
        # fix_subnet is not None, means subnet retraining.
        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self.architecture, fix_subnet)
            self.is_supernet = False

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE) -> BaseMutator:
        """build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, BaseMutator):
            raise TypeError('mutator should be a `dict` or '
                            '`OneShotModuleMutator` instance, but got '
                            f'{type(mutator)}')

        return mutator

    def _build_distiller(
            self, distiller: VALID_DISTILLER_TYPE) -> ConfigurableDistiller:
        if isinstance(distiller, dict):
            distiller = MODELS.build(distiller)
        if not isinstance(distiller, ConfigurableDistiller):
            raise TypeError('distiller should be a `dict` or '
                            '`ConfigurableDistiller` instance, but got '
                            f'{type(distiller)}')

        return distiller

    def sample_subnet(self) -> Dict:
        """Random sample subnet by mutator."""
        subnet_dict = dict()
        for name, mutator in self.mutators.items():
            if name == 'value_mutator':
                subnet_dict.update(
                    dict((str(group_id), value) for group_id, value in
                         mutator.sample_choices().items()))
            else:
                subnet_dict.update(mutator.sample_choices())
        return subnet_dict

    def set_subnet(self, subnet_dict: Dict) -> None:
        """Set the subnet sampled by :meth:sample_subnet."""
        for name, mutator in self.mutators.items():
            if name == 'value_mutator':
                value_subnet = dict((int(group_id), value)
                                    for group_id, value in subnet_dict.items()
                                    if isinstance(group_id, str))
                mutator.set_choices(value_subnet)
            else:
                channel_subnet = dict(
                    (group_id, value)
                    for group_id, value in subnet_dict.items()
                    if isinstance(group_id, int))
                mutator.set_choices(channel_subnet)

    def set_max_subnet(self) -> None:
        """Set max subnet."""
        for mutator in self.mutators.values():
            mutator.set_choices(mutator.max_choices)

    def set_min_subnet(self) -> None:
        """Set min subnet."""
        for mutator in self.mutators.values():
            mutator.set_choices(mutator.min_choices)

    @property
    def search_groups(self) -> Dict:
        search_groups = dict()

        for name, mutator in self.mutators.items():
            search_groups[name] = mutator.search_groups

        return search_groups

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        selects: List = self.strategy_groups[self.strategy]

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            subnet_losses = dict()
            with optim_wrapper.optim_context(
                    self), self.distiller.student_recorders:  # type: ignore
                _ = self(batch_inputs, data_samples, mode='loss')
                soft_loss = self.distiller.compute_distill_losses()

                subnet_losses.update(soft_loss)

                parsed_subnet_losses, _ = self.parse_losses(subnet_losses)
                optim_wrapper.update_params(parsed_subnet_losses)

            return subnet_losses

        if self.strategy == 'sandwich4':
            if not self._optim_wrapper_count_status_reinitialized:
                reinitialize_optim_wrapper_count_status(
                    model=self,
                    optim_wrapper=optim_wrapper,
                    accumulative_counts=self.num_samples + 2)
                self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.data_preprocessor(data,
                                                            True).values()

        total_losses = dict()

        for kind in selects:
            if kind in ('max'):
                self.set_max_subnet()
                self.architecture.backbone.set_dropout(self.drop_prob)
                with optim_wrapper.optim_context(
                        self
                ), self.distiller.teacher_recorders:  # type: ignore
                    max_subnet_losses = self(
                        batch_inputs, data_samples, mode='loss')
                    parsed_max_subnet_losses, _ = self.parse_losses(
                        max_subnet_losses)
                    optim_wrapper.update_params(parsed_max_subnet_losses)
                total_losses.update(
                    add_prefix(max_subnet_losses, 'max_subnet'))
            elif kind in ('min'):
                self.set_min_subnet()
                self.architecture.backbone.set_dropout(0.)
                min_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(min_subnet_losses, 'min_subnet'))
            elif kind in ('random', 'random1'):
                for sample_idx in range(self.num_samples):
                    self.set_subnet(self.sample_subnet())
                    random_subnet_losses = distill_step(
                        batch_inputs, data_samples)
                    total_losses.update(
                        add_prefix(random_subnet_losses,
                                   f'random_subnet_{sample_idx}'))

        return total_losses


@MODEL_WRAPPERS.register_module()
class BigNASDDP(MMDistributedDataParallel):

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            subnet_losses = dict()
            with optim_wrapper.optim_context(
                    self
            ), self.module.distiller.student_recorders:  # type: ignore
                _ = self(batch_inputs, data_samples, mode='loss')
                soft_loss = self.module.distiller.compute_distill_losses()

                subnet_losses.update(soft_loss)

                parsed_subnet_losses, _ = self.module.parse_losses(
                    subnet_losses)
                optim_wrapper.update_params(parsed_subnet_losses)

            return subnet_losses

        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=self.module.num_samples + 2)
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.module.data_preprocessor(
            data, True).values()

        total_losses = dict()
        self.module.set_max_subnet()
        # TODO
        self.module.architecture.backbone.set_dropout(self.module.drop_prob)
        with optim_wrapper.optim_context(
                self), self.module.distiller.teacher_recorders:  # type: ignore
            max_subnet_losses = self(batch_inputs, data_samples, mode='loss')
            parsed_max_subnet_losses, _ = self.module.parse_losses(
                max_subnet_losses)
            optim_wrapper.update_params(parsed_max_subnet_losses)
        total_losses.update(add_prefix(max_subnet_losses, 'max_subnet'))

        self.module.set_min_subnet()
        # TODO
        self.module.architecture.backbone.set_dropout(0.)
        min_subnet_losses = distill_step(batch_inputs, data_samples)
        total_losses.update(add_prefix(min_subnet_losses, 'min_subnet'))

        for sample_idx in range(self.module.num_samples):
            self.module.set_subnet(self.module.sample_subnet())
            random_subnet_losses = distill_step(batch_inputs, data_samples)
            total_losses.update(
                add_prefix(random_subnet_losses,
                           f'random_subnet_{sample_idx}'))

        return total_losses

    @property
    def _optim_wrapper_count_status_reinitialized(self) -> bool:
        return self.module._optim_wrapper_count_status_reinitialized

    @_optim_wrapper_count_status_reinitialized.setter
    def _optim_wrapper_count_status_reinitialized(self, val: bool) -> None:
        assert isinstance(val, bool)

        self.module._optim_wrapper_count_status_reinitialized = val
