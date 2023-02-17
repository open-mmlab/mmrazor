# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.registry import MODELS

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class BaseAlgorithm(BaseModel):
    """Base class for algorithms.

    BaseAlgorithm inherit from BaseModel. BaseModel implements the basic
    functions of the algorithmic model, such as weights initialize,
    batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.
    More details of BaseModel could see docs for :class:`BaseModel`.

    :obj:`BaseAlgorithm` forward just is a wrapper of :obj:`BaseModel` forward.
    Various compression algorithms can be implemented by inheriting
    BaseAlgorithm.

    Subclasses inherit from BaseAlgorithm only need to override the
    :meth:`loss`, which implements the logic to calculate loss, then
    can be trained in the runner.

    Args:
        architecture (dict | :obj:`BaseModel`): The config of
            :class:`BaseModel` or built model.
        data_preprocessor (dict | torch.nn.Module | None): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (dict): The weight initialized config for
            :class:`BaseModule`.
        module_inplace(bool): Whether to allow module inplace attribute True.
            Defaults to False.

    Note:
        If `data_preprocessor` is None, :obj:`BaseAlgorithm` will set
        `data_preprocessor` to model's `data_preprocessor`.


    Attributes:
        architecture (:obj:`BaseModel`): Model that needs to be compressed.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None,
                 module_inplace: bool = False) -> None:

        # super().__init__() needs built data_preprocessor, so
        # build model first.
        if isinstance(architecture, Dict):
            architecture = MODELS.build(architecture)

        if not isinstance(architecture, BaseModel):
            raise TypeError('architecture should be a `dict` or '
                            f'`BaseModel` instance, but got '
                            f'{type(architecture)}')

        # If `data_preprocessor` is None, there will set
        # `data_preprocessor` to model's `data_preprocessor`.
        if data_preprocessor is None:
            # use model's data_preprocessor
            data_preprocessor = getattr(architecture, 'data_preprocessor',
                                        None)
        super().__init__(data_preprocessor, init_cfg)

        # Cannot assign module before Module.__init__()
        self.architecture = architecture

        # Find all nn.Modules in the model that contain the 'inplace' attribute
        # and set them to False
        self.module_inplace = module_inplace
        if not self.module_inplace:
            self.set_module_inplace_false(architecture, 'self.architecture')
        pass

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``batch_inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``
                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:
                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict of tensor for custom use.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'predict':
            return self._predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""
        return self.architecture(inputs, data_samples, mode='loss')

    def _forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> TensorResults:
        """Network forward process."""
        return self.architecture(inputs, data_samples, mode='tensor')

    def _predict(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> PredictResults:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        return self.architecture(inputs, data_samples, mode='predict')

    def set_module_inplace_false(self, architecture: Union[OrderedDict,
                                                           nn.Module],
                                 varstr: str) -> None:
        """Find all nn.Modules in the model that contain the 'inplace'
        attribute and set them to False in order to prevent occur error in
        Recorders using recursion algorithm.

        This function will disassemble the Args architecture .If type
        'nn.Module' is detected, determine if it contains an 'inplace'
        attribute and set False if it does. If none, get the OrderedDict
        and then iterate through the dictionary to continue the recursive
        search.

        Args:
            architecture (OrderedDict | nn.Module): The config OrderedDict
            for model or built model.
            varstr (str): Records the call-level string containing the
            'inplace' attribute.

        Returns:
            None
        """

        if isinstance(architecture, nn.Module):
            if hasattr(eval(varstr), 'inplace'):
                eval(varstr).inplace = False
            else:
                self.set_module_inplace_false(architecture._modules,
                                              varstr + '._modules')
        elif isinstance(architecture, OrderedDict):
            for key, value in architecture.items():
                self.set_module_inplace_false(value, varstr + f"['{key}']")
        else:
            return
