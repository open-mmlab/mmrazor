# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ..base import BaseAlgorithm, BaseModel

try:
    from torch.ao.quantization import FakeQuantizeBase
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class MMArchitectureQuant(BaseAlgorithm):
    """General quantization.

    Args:
        architecture (Union[Dict, BaseModel]): The config of model to be
            quantized.
        quantizer (Union[Dict, BaseModel]): The quantizer to support different
            backend type.
        qmodel_modes (List): The available mode of runner.
        data_preprocessor (Optional[Dict]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        forward_modes (Tuple): The modes in forward method in OpenMMLab
            architecture could be tensor, predict, or loss. It can generate
            different graph of quantized model.
        float_checkpoint (Optional[str]): The path of pretrained FP checkpoint.
            Quantization is different from or task, we recommend to use
            `float_checkpoint` as pretrain model. Defaults to None.
        init_cfg (Optional[Dict]): The weight initialized config for:
            class:`BaseModule`.

    Note:
        forward_modes (Tuple): In OpenMMLab architecture, differenet modes
            will trace a different graph of quantized model.
    """

    def __init__(self,
                 architecture: Union[Dict, BaseModel],
                 quantizer: Union[Dict, BaseModel],
                 data_preprocessor: Optional[Dict] = None,
                 forward_modes: Tuple = ('tensor', 'predict', 'loss'),
                 float_checkpoint: Optional[str] = None,
                 input_shapes: Tuple = (1, 3, 224, 224),
                 init_cfg: Optional[Dict] = None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        # Default to mmcls.ClsDataPreprocessor.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')
        super().__init__(architecture, data_preprocessor, init_cfg)
        # If we have a float_checkpoint, we load it as pretrain.
        if float_checkpoint:
            _ = load_checkpoint(self.architecture, float_checkpoint)
            self.architecture._is_init = True

        self.quantizer = MODELS.build(quantizer)
        self.input_shapes = input_shapes
        self.forward_modes = forward_modes

        self.qmodels = self._build_qmodels(self.architecture)

        self.sync_qparams('predict')

    def sync_qparams(self, src_mode: str):
        """Sync all quantize parameters in different `forward_modes`. We could
        have more than one forward mode to generate graphs, each mode will
        generate one graph. But in training, only one graph will be update, so
        we need to sync qparams in the other graphs.

        Args:
            src_mode (str): The modes of forward method.

        Note:
            `traverse()` function recursively traverses all module to sync
                quantized graph generated from different `forward_modes`.
                This is because We have different mode ('tensor', 'predict',
                'loss') in OpenMMLab architecture which have different graph
                in some subtle ways, so we need to sync them here.
        """

        def traverse(module, prefix):
            for name, child in module._modules.items():
                if module is None:
                    continue
                child_name = f'{prefix}{name}'
                if isinstance(child, FakeQuantizeBase):
                    for name, param in child.named_parameters():
                        param_name = f'{child_name}.{name}'
                        src_param = src_state_dict[param_name]
                        if src_param.shape == param.shape:
                            param.data.copy_(src_param)
                        else:
                            # requirs_grad = param.requires_grad
                            # param.requires_grad = False
                            param.resize_(src_param.shape)
                            # param.requires_grad = requirs_grad
                            param.data.copy_(src_param)
                    for name, buffer in child.named_buffers():
                        buffer_name = f'{child_name}.{name}'
                        src_buffer = src_state_dict[buffer_name]
                        if src_buffer.shape == buffer.shape:
                            buffer.data.copy_(src_buffer)
                        else:
                            buffer.resize_(src_buffer.shape)
                            buffer.data.copy_(src_buffer)
                else:
                    traverse(child, f'{child_name}.')

        src_state_dict = self.qmodels[src_mode].state_dict()
        for mode in self.forward_modes:
            if mode == src_mode:
                continue
            traverse(self.qmodels[mode], '')

    def _build_qmodels(self, model: BaseModel):
        """Build quantized models from the given model.

        Args:
            model (BaseModel): the given fp model.

        Example:
            The main body of the graph is all the same, but the last one or two
            op will have difference, as shown below.

            self.qmodels['tensor'].graph.print_tabular()
            opcode       target            args
            call_module  head.fc           (activation_post_process_38,)
            output       output            (head_fc,)

            self.qmodels['loss'].graph.print_tabular()
            opcode       target            args
            call_method  _get_loss         (head, head_fc, data_samples)
            output       output            (_get_loss,)

            self.qmodels['predict'].graph.print_tabular()
            opcode       target            args
            call_method  _get_predictions  (head, head_fc, data_samples)
            output       output            (_get_predictions,)
        """

        qmodels = nn.ModuleDict()
        for mode in self.forward_modes:
            concrete_args = {'mode': mode}
            observed_module = self.quantizer.prepare(model, concrete_args)
            qmodels[mode] = observed_module

        return qmodels

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode in self.qmodels:
            qmodel = self.qmodels[mode]
            return qmodel(inputs, data_samples, mode)
        else:
            return self.architecture(inputs, data_samples, mode)

    def calibrate_step(self, data: Union[Dict, Tuple, List]):
        """PTQ method need calibrate by cali data."""

        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')


@MODEL_WRAPPERS.register_module()
class MMArchitectureQuantDDP(MMDistributedDataParallel):
    """DDPwapper for GeneralQuant.

    Args:
        device_ids (Optional[Union[List, int, torch.device]]): devices to run
        ddp.
    """

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:

        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)
        # After moving all model parameters and buffers to the GPU
        # (`model.cuda()`), the buffers in model are different.
        self.module.qmodels = self.module._build_qmodels(
            self.module.architecture)

    def calibrate_step(self, data: Union[Dict, Tuple, List]):
        """PTQ method need calibrate by cali data."""

        return self.module.calibrate_step(data)

    def sync_qparams(self, src: str):
        """Same as in 'MMArchitectureQuant'. Sync all quantize parameters in
        different `forward_modes`. We could have several modes to generate
        graphs, but in training, only one graph will be update, so we need to
        sync qparams on the other graphs.

        Args:
            src (str): The src modes of forward method.

        Note:
            `traverse()` function recursively traverses all module to sync
                quantized graph generated from different `forward_modes`.
                This is because We have different mode ('tensor', 'predict',
                'loss') in OpenMMLab architecture which have different graph
                in some subtle ways, so we need to sync them here.
        """

        self.module.sync_qparams(src)
