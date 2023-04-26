# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import Config
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.utils import pop_rewriter_function_record
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.structures.quantization import QConfigHandler
from ..base import BaseAlgorithm, BaseModel

try:
    from torch.ao.quantization import (FakeQuantizeBase, MinMaxObserver,
                                       PerChannelMinMaxObserver,
                                       disable_observer)
except ImportError:
    from mmrazor.utils import get_placeholder

    FakeQuantizeBase = get_placeholder('torch>=1.13')
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')
    disable_observer = get_placeholder('torch>=1.13')

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class MMArchitectureQuant(BaseAlgorithm):
    """General quantization for OpenMMLab's models.

    Args:
        architecture (Union[Dict, BaseModel]): The config of model to be
            quantized.
        quantizer (Union[Dict, BaseModel]): The quantizer to support different
            backend type.
        deploy_cfg (Union[str, Dict]): Deployment config file or Config object.
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
                 deploy_cfg: Optional[Union[str, Dict]] = None,
                 data_preprocessor: Optional[Dict] = None,
                 forward_modes: Tuple = ('tensor', 'predict', 'loss'),
                 float_checkpoint: Optional[str] = None,
                 input_shapes: Tuple = (1, 3, 224, 224),
                 init_cfg: Optional[Dict] = None):

        super().__init__(architecture, data_preprocessor, init_cfg)

        self.quantizer = MODELS.build(quantizer)
        self.input_shapes = input_shapes
        self.forward_modes = forward_modes
        if isinstance(deploy_cfg, str):
            deploy_cfg = Config.fromfile(deploy_cfg)
        self.deploy_cfg = deploy_cfg

        # Replace syncbn and _BatchNormXd (in mmengine) with batchnorm2d
        self.quantizer.convert_batchnorm2d(self.architecture)

        # If we have a float_checkpoint, we load it as pretrain.
        if float_checkpoint:
            _ = load_checkpoint(self.architecture, float_checkpoint)
            self.architecture._is_init = True

        self.qmodels = self._build_qmodels(self.architecture)
        self.sync_qparams('tensor')
        self.reset_observer_and_fakequant_statistics(self)

    def reset_observer_and_fakequant_statistics(self, model):
        """Reset the statistics in observers and fake quantizers.

        The forward computation in `_build_qmodels` can modify the original
        statistics in observers and fake quantizers.
        """
        for module in model.modules():
            if isinstance(module, (MinMaxObserver, PerChannelMinMaxObserver)):
                module.reset_min_max_vals()
            elif isinstance(module, FakeQuantizeBase):
                module.scale.data = torch.ones_like(module.scale)
                module.zero_point.data = torch.zeros_like(module.zero_point)

    def sync_qparams(self, src_mode: str):
        """Sync all quantize parameters in different `forward_modes`. We could
        have more than one forward mode to generate graphs, each mode will
        generate one graph. But in training, only one graph will be update, so
        we need to sync qparams in the other graphs.

        Args:
            src_mode (str): The modes of forward method.

        Note:
            `traverse()` method recursively traverses all modules to sync
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
                            requirs_grad = param.requires_grad
                            param.requires_grad = False
                            param.resize_(src_param.shape)
                            param.requires_grad = requirs_grad
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

    def _get_rewriter_context_in_mmdeploy(self, deploy_cfg):
        """Get rewriter context in mmdeploy according to the deploy related
        config."""
        from mmdeploy.apis.onnx.passes import optimize_onnx
        from mmdeploy.codebase import import_codebase
        from mmdeploy.core import RewriterContext
        from mmdeploy.utils import (IR, Backend, get_backend, get_codebase,
                                    get_dynamic_axes, get_ir_config,
                                    get_onnx_config)
        from mmdeploy.utils.config_utils import get_codebase_external_module

        codebase = get_codebase(deploy_cfg)
        custom_module_list = get_codebase_external_module(deploy_cfg)
        import_codebase(codebase, custom_module_list)

        def _add_or_update(cfg: dict, key: str, val: Any):
            if key in cfg and isinstance(cfg[key], dict) and isinstance(
                    val, dict):
                cfg[key].update(val)
            else:
                cfg[key] = val

        context_info = dict()
        deploy_cfg = copy.deepcopy(deploy_cfg)

        backend = get_backend(deploy_cfg).value

        onnx_cfg = get_onnx_config(deploy_cfg)
        opset_version = onnx_cfg.get('opset_version', 11)

        input_names = onnx_cfg['input_names']
        output_names = onnx_cfg['output_names']
        axis_names = input_names + output_names
        dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)

        verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get(
            'verbose', False)
        keep_initializers_as_inputs = onnx_cfg.get(
            'keep_initializers_as_inputs', True)
        optimize = onnx_cfg.get('optimize', False)
        if backend == Backend.NCNN.value:
            """NCNN backend needs a precise blob counts, while using onnx
            optimizer will merge duplicate initilizers without reference
            count."""
            optimize = False

        ir_config = dict(
            type='onnx',
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs)

        _add_or_update(deploy_cfg, 'ir_config', ir_config)
        ir = IR.get(get_ir_config(deploy_cfg)['type'])
        if isinstance(backend, Backend):
            backend = backend.value
        backend_config = dict(type=backend)
        _add_or_update(deploy_cfg, 'backend_config', backend_config)

        context_info['cfg'] = deploy_cfg
        context_info['ir'] = ir
        if 'backend' not in context_info:
            context_info['backend'] = backend
        if 'opset' not in context_info:
            context_info['opset'] = opset_version

        if 'onnx_custom_passes' not in context_info:
            onnx_custom_passes = optimize_onnx if optimize else None
            context_info['onnx_custom_passes'] = onnx_custom_passes

        return RewriterContext(**context_info)

    def _pop_function_record_in_rewriter_context(self, rewriter_context):
        """Delete user-specific rewriters from
        `RewriterContext._rewriter_manager`. We use the model which is
        rewritten by mmdeploy to build quantized models. However not all the
        functions rewritten by mmdeploy need to be rewritten in mmrazor. For
        example, mmdeploy rewrite
        `mmcls.models.classifiers.ImageClassifier.forward` and
        `mmcls.models.classifiers.BaseClassifier.forward` for deployment. But
        they can't be rewritten by mmrazor as ptq and qat are done in mmrazor.
        So to ensure ptq and qat proceed normally, we have to remove these
        record from `RewriterContext._rewriter_manager`.

        Args:
            rewriter_context (RewriterContext): The RewriterContext used in
                mmdeploy.
        """
        skipped_methods = getattr(self.quantizer.tracer, 'skipped_methods', [])
        function_record_to_pop = self.deploy_cfg.get('function_record_to_pop',
                                                     [])
        function_record_to_pop.extend(skipped_methods)
        return pop_rewriter_function_record(rewriter_context,
                                            function_record_to_pop)

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

        rewriter_context = self._get_rewriter_context_in_mmdeploy(
            self.deploy_cfg) if self.deploy_cfg is not None else None

        if rewriter_context is not None:
            # Pop function records in `quantizer.tracer.skipped_method`
            # temporarily
            function_record_backup = \
                self._pop_function_record_in_rewriter_context(rewriter_context)

        qmodels = nn.ModuleDict()
        for mode in self.forward_modes:
            concrete_args = {'mode': mode}

            if rewriter_context is not None:
                with rewriter_context:
                    observed_module = self.quantizer.prepare(
                        model, concrete_args)
            else:
                observed_module = self.quantizer.prepare(model, concrete_args)

            qmodels[mode] = observed_module

        if rewriter_context is not None:
            # Add these popped function records back.
            rewriter_context._rewriter_manager.function_rewriter. \
                _registry._rewrite_records.update(function_record_backup)

        # data_samples can not be None in detectors during prediction.
        # But we need to make the dummy prediction in _build_qmodels.
        # It is more convenient to use `tensor` mode.
        is_training = qmodels['tensor'].training
        # Avoid random input changing bn's statistics
        qmodels['tensor'].eval()
        # Originally, the steps to train a qat model is as follows:
        # 1. build qmodels 2. convert the model to ddpmodel 3. forward backward
        # The shape of `scale` and `zero_point` can be modified during forward.
        # We initialize these parameters with per-tensor mode by default for
        # convenience. Their shape will be modified during forward if
        # per-channel mode is used. It's hacky. Hence we need to input a
        # dummy input to make sure the shape has been modified.
        device = next(qmodels.parameters()).device
        dummy_input = torch.randn(self.input_shapes).to(device)
        qmodels['tensor'](dummy_input, None, 'tensor')
        qmodels['tensor'].train(mode=is_training)

        return qmodels

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward with qmodels in quantization."""

        if mode in self.qmodels:
            qmodel = self.qmodels[mode]
            return qmodel(inputs, data_samples, mode)
        else:
            return self.architecture(inputs, data_samples, mode)

    def calibrate_step(self, data: Union[Dict, Tuple, List]):
        """PTQ method need calibrate by cali data."""

        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')

    def get_deploy_model(self):
        """Prepare for deploy to the backend with mmdeploy, which will be used
        in mmdeploy, and usually includes as follows:

        1. prepare for the float model rewritten by mmdeploy.
        2. load checkpoint consists of float weight and quantized params in
        mmrazor.
        3. post process weight fakequant for exporting .onnx that meet
        the backend's requirement.
        """
        device = next(self.parameters()).device
        quantized_state_dict = self.qmodels['predict'].state_dict()
        fp32_model = self.architecture
        self.quantizer.convert_batchnorm2d(fp32_model)
        observed_model = self.quantizer.prepare(fp32_model)
        observed_model.load_state_dict(quantized_state_dict)

        self.quantizer.post_process_for_deploy(
            observed_model,
            device=device,
            keep_w_fake_quant=True,
            update_weight_with_fakequant=True)

        # replace various activation fakequant with base fakequant, which
        # contributes to deploy our model to various backends.
        for node in observed_model.graph.nodes:
            if 'activation_post_process_' in node.name:
                module_name = node.target
                module = getattr(observed_model, module_name)
                fakequant_new = QConfigHandler.replace_fakequant(
                    module,
                    self.quantizer.qconfig.a_qscheme,
                    update_qparams=True)
                setattr(observed_model, module_name, fakequant_new)

        observed_model.apply(disable_observer)

        return observed_model


@MODEL_WRAPPERS.register_module()
class MMArchitectureQuantDDP(MMDistributedDataParallel):
    """DDPwapper for MMArchitectureQuant.

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
        self.module.sync_qparams('tensor')
        self.module.reset_observer_and_fakequant_statistics(self)

    def calibrate_step(self, data: Union[Dict, Tuple, List]):
        """PTQ method need calibrate by cali data."""

        return self.module.calibrate_step(data)

    def sync_qparams(self, src_mode: str):
        """Same as in 'MMArchitectureQuant'. Sync all quantize parameters in
        different `forward_modes`. We could have several modes to generate
        graphs, but in training, only one graph will be update, so we need to
        sync qparams on the other graphs.

        Args:
            src_mode (str): The src modes of forward method.
        """

        self.module.sync_qparams(src_mode)
