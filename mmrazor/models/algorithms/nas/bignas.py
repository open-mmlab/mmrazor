# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
from mmengine import BaseDataElement
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from torch import nn

from mmrazor.models.architectures.dynamic_op import DynamicInputResizer
from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[BaseMutator, Dict]
VALID_MUTATORS_TYPE = Dict[str, Union[BaseMutator, Dict]]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class BigNAS(BaseAlgorithm):

    def __init__(self,
                 mutators: VALID_MUTATORS_TYPE,
                 distiller: VALID_DISTILLER_TYPE,
                 architecture: Union[BaseModel, Dict],
                 resizer_cfg: Optional[Dict] = None,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None,
                 num_samples: int = 2) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        built_mutators = dict()
        for name, mutator in mutators.items():
            mutator = self._build_mutator(mutator)
            built_mutators[name] = mutator
            # `prepare_from_supernet` must be called before distiller initialized  # noqa: E501
            mutator.prepare_from_supernet(self.architecture)
        self.mutators = built_mutators

        self.distiller = self._build_distiller(distiller)
        self.distiller.prepare_from_teacher(self.architecture)
        self.distiller.prepare_from_student(self.architecture)

        self.input_resizer = self._build_input_resizer(resizer_cfg)

        self.num_samples = num_samples

        self._optim_wrapper_count_status_reinitialized = False

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE) -> BaseMutator:
        """build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, BaseMutator):
            raise TypeError('mutator should be a `dict` or '
                            '`OneShotModuleMutator` instance, but got '
                            f'{type(mutator)}')

        return mutator

    def _build_input_resizer(
            self,
            resizer_cfg: Optional[Dict]) -> Optional[DynamicInputResizer]:
        if resizer_cfg is None:
            return None
        input_resizer_cfg = resizer_cfg['input_resizer']
        input_resizer = MODELS.build(input_resizer_cfg)
        if not isinstance(input_resizer, DynamicInputResizer):
            raise TypeError('input_resizer should be a `dict` or '
                            '`DynamicInputResizer` instance, but got '
                            f'{type(input_resizer)}')
        mutable_shape_cfg = resizer_cfg['mutable_shape']
        mutable_shape = MODELS.build(mutable_shape_cfg)
        if not isinstance(mutable_shape, OneShotMutableValue):
            raise ValueError('input_resizer should have OneShotMutableValue '
                             'attribute `mutable_depth`.')
        input_resizer.mutate_shape(mutable_shape)

        return input_resizer

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
        subnet_dict = dict()

        for name, mutator in self.mutators.items():
            subnet_dict[name] = mutator.sample_choices()
        if self.input_resizer is not None:
            random_choice = self.input_resizer.sample_choice()
            subnet_dict['input_resizer'] = random_choice

        return subnet_dict

    def set_subnet(self, subnet_dict: Dict) -> None:
        for name, mutator in self.mutators.items():
            mutator.set_choices(subnet_dict[name])
        if self.input_resizer is not None:
            self.input_resizer.current_choice = \
                subnet_dict['input_resizer']

    def set_max_subnet(self) -> None:
        for mutator in self.mutators.values():
            mutator.set_choices(mutator.max_choices)
        if self.input_resizer is not None:
            max_choice = self.input_resizer.max_choice
            self.input_resizer.current_choice = max_choice

    def set_min_subnet(self) -> None:
        for mutator in self.mutators.values():
            mutator.set_choices(mutator.min_choices)
        if self.input_resizer is not None:
            min_choice = self.input_resizer.min_choice
            self.input_resizer.current_choice = min_choice

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            subnet_losses = dict()
            with optim_wrapper.optim_context(
                    self), self.distiller.student_recorders:  # type: ignore
                hard_loss = self(batch_inputs, data_samples, mode='loss')
                soft_loss = self.distiller.compute_distill_losses()

                subnet_losses.update(hard_loss)
                subnet_losses.update(soft_loss)

                parsed_subnet_losses, _ = self.parse_losses(subnet_losses)
                optim_wrapper.update_params(parsed_subnet_losses)

            return subnet_losses

        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=self.num_samples + 2)
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.data_preprocessor(data, True)
        if self.input_resizer is not None:
            batch_inputs = self.input_resizer(batch_inputs)

        total_losses = dict()
        self.set_max_subnet()
        with optim_wrapper.optim_context(
                self), self.distiller.teacher_recorders:  # type: ignore
            max_subnet_losses = self(batch_inputs, data_samples, mode='loss')
            parsed_max_subnet_losses, _ = self.parse_losses(max_subnet_losses)
            optim_wrapper.update_params(parsed_max_subnet_losses)
        total_losses.update(add_prefix(max_subnet_losses, 'max_subnet'))

        self.set_min_subnet()
        min_subnet_losses = distill_step(batch_inputs, data_samples)
        total_losses.update(add_prefix(min_subnet_losses, 'min_subnet'))

        for sample_idx in range(self.num_samples):
            self.set_subnet(self.sample_subnet())
            random_subnet_losses = distill_step(batch_inputs, data_samples)
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
                hard_loss = self(batch_inputs, data_samples, mode='loss')
                soft_loss = self.module.distiller.compute_distill_losses()

                subnet_losses.update(hard_loss)
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

        batch_inputs, data_samples = self.module.data_preprocessor(data, True)
        if self.module.input_resizer is not None:
            batch_inputs = self.module.input_resizer(batch_inputs)

        total_losses = dict()
        self.module.set_max_subnet()
        with optim_wrapper.optim_context(
                self), self.module.distiller.teacher_recorders:  # type: ignore
            max_subnet_losses = self(batch_inputs, data_samples, mode='loss')
            parsed_max_subnet_losses, _ = self.module.parse_losses(
                max_subnet_losses)
            optim_wrapper.update_params(parsed_max_subnet_losses)
        total_losses.update(add_prefix(max_subnet_losses, 'max_subnet'))

        self.module.set_min_subnet()
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
