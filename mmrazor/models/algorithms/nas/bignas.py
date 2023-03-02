# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.architectures.ops.mobilenet_series import MBBlock
from mmrazor.models.architectures.utils import set_dropout
from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators import NasMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class BigNAS(BaseAlgorithm):
    """Implementation of `BigNas <https://arxiv.org/pdf/2003.11142>`_

    BigNAS is a NAS algorithm which searches the following items in MobileNetV3
    with the one-shot paradigm: kernel_sizes, out_channels, expand_ratios,
    block_depth and input sizes.

    BigNAS uses a `sandwich` strategy to sample subnets from the supernet,
    which includes the max subnet, min subnet and N random subnets. It doesn't
    require retraining, therefore we can directly get well-trained subnets
    after supernet training.

    The logic of the search part is implemented in
    :class:`mmrazor.engine.EvolutionSearchLoop`

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (VALID_MUTATOR_TYPE): The config of :class:`NasMutator` or
            built mutator.
        distiller (VALID_DISTILLER_TYPE): Cfg of :class:`ConfigurableDistiller`
            or built distiller.
        data_preprocessor (Optional[Union[dict, nn.Module]]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        num_random_samples (int): number of random sample subnets.
            Defaults to 2.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.2.
        backbone_dropout_stages (List): Stages to be set dropout. Defaults to
            [6, 7].
        init_cfg (Optional[dict]): Init config for ``BaseModule``.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE = None,
                 distiller: VALID_DISTILLER_TYPE = None,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 num_random_samples: int = 2,
                 drop_path_rate: float = 0.2,
                 backbone_dropout_stages: List = [6, 7],
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator = self._build_mutator(mutator)
        # NOTE: `mutator.prepare_from_supernet` must be called
        # before distiller initialized.
        self.mutator.prepare_from_supernet(self.architecture)

        self.distiller = self._build_distiller(distiller)
        self.distiller.prepare_from_teacher(self.architecture)
        self.distiller.prepare_from_student(self.architecture)

        self.sample_kinds = ['max', 'min']
        for i in range(num_random_samples):
            self.sample_kinds.append('random' + str(i))

        self.drop_path_rate = drop_path_rate
        self.backbone_dropout_stages = backbone_dropout_stages
        self._optim_wrapper_count_status_reinitialized = False

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE = None) -> NasMutator:
        """Build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, NasMutator):
            raise TypeError('mutator should be a `dict` or `NasMutator` '
                            f'instance, but got {type(mutator)}.')
        return mutator

    def _build_distiller(
            self,
            distiller: VALID_DISTILLER_TYPE = None) -> ConfigurableDistiller:
        """Build distiller."""
        if isinstance(distiller, dict):
            distiller = MODELS.build(distiller)
        if not isinstance(distiller, ConfigurableDistiller):
            raise TypeError('distiller should be a `dict` or '
                            '`ConfigurableDistiller` instance, but got '
                            f'{type(distiller)}')
        return distiller

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

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

        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=len(self.sample_kinds))
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.data_preprocessor(data,
                                                            True).values()

        total_losses = dict()
        for kind in self.sample_kinds:
            # update the max subnet loss.
            if kind == 'max':
                self.mutator.set_max_choices()
                set_dropout(
                    layers=self.architecture.backbone.layers[:-1],
                    module=MBBlock,
                    dropout_stages=self.backbone_dropout_stages,
                    drop_path_rate=self.drop_path_rate)
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
            # update the min subnet loss.
            elif kind == 'min':
                self.mutator.set_min_choices()
                set_dropout(
                    layers=self.architecture.backbone.layers[:-1],
                    module=MBBlock,
                    dropout_stages=self.backbone_dropout_stages,
                    drop_path_rate=0.)
                min_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(min_subnet_losses, 'min_subnet'))
            # update the random subnets loss.
            elif 'random' in kind:
                self.mutator.set_choices(self.mutator.sample_choices())
                set_dropout(
                    layers=self.architecture.backbone.layers[:-1],
                    module=MBBlock,
                    dropout_stages=self.backbone_dropout_stages,
                    drop_path_rate=0.)
                random_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(random_subnet_losses, f'{kind}_subnet'))

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                accumulative_counts=len(self.module.sample_kinds))
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.module.data_preprocessor(
            data, True).values()

        total_losses = dict()
        for kind in self.module.sample_kinds:
            # update the max subnet loss.
            if kind == 'max':
                self.module.mutator.set_max_choices()
                set_dropout(
                    layers=self.module.architecture.backbone.layers[:-1],
                    module=MBBlock,
                    dropout_stages=self.module.backbone_dropout_stages,
                    drop_path_rate=self.module.drop_path_rate)
                with optim_wrapper.optim_context(
                        self
                ), self.module.distiller.teacher_recorders:  # type: ignore
                    max_subnet_losses = self(
                        batch_inputs, data_samples, mode='loss')
                    parsed_max_subnet_losses, _ = self.module.parse_losses(
                        max_subnet_losses)
                    optim_wrapper.update_params(parsed_max_subnet_losses)
                total_losses.update(
                    add_prefix(max_subnet_losses, 'max_subnet'))
            # update the min subnet loss.
            elif kind == 'min':
                self.module.mutator.set_min_choices()
                set_dropout(
                    layers=self.module.architecture.backbone.layers[:-1],
                    module=MBBlock,
                    dropout_stages=self.module.backbone_dropout_stages,
                    drop_path_rate=0.)
                min_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(min_subnet_losses, 'min_subnet'))
            # update the random subnets loss.
            elif 'random' in kind:
                self.module.mutator.set_choices(
                    self.module.mutator.sample_choices())
                set_dropout(
                    layers=self.module.architecture.backbone.layers[:-1],
                    module=MBBlock,
                    dropout_stages=self.module.backbone_dropout_stages,
                    drop_path_rate=0.)
                random_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(random_subnet_losses, f'{kind}_subnet'))

        return total_losses

    @property
    def _optim_wrapper_count_status_reinitialized(self) -> bool:
        return self.module._optim_wrapper_count_status_reinitialized

    @_optim_wrapper_count_status_reinitialized.setter
    def _optim_wrapper_count_status_reinitialized(self, val: bool) -> None:
        assert isinstance(val, bool)

        self.module._optim_wrapper_count_status_reinitialized = val
