# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmengine.config import Config

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer)
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
    from torch.ao.quantization.qconfig_mapping import (
        _FIXED_QPARAMS_OP_TO_OBSERVER, FixedQParamsFakeQuantize, QConfig,
        QConfigMapping, default_weight_fake_quant)
    from torch.ao.quantization.quantize_fx import _fuse_fx
    from torch.fx.graph_module import GraphModule
    from torch.nn.intrinsic.qat import modules as qat_fused_modules
    from torch.nn.qat import modules as qat_modules
    from torch.onnx import register_custom_op_symbolic
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    GraphModule = get_placeholder('torch>=1.13')
    ObservedGraphModule = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    disable_observer = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')
    qat_fused_modules = get_package_placeholder('torch>=1.13')
    qat_modules = get_package_placeholder('torch>=1.13')
    _FIXED_QPARAMS_OP_TO_OBSERVER = get_package_placeholder('torch>=1.13')
    FixedQParamsFakeQuantize = get_package_placeholder('torch>=1.13')
    QConfig = get_package_placeholder('torch>=1.13')
    default_weight_fake_quant = get_package_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.models.task_modules.tracer import build_graphmodule
from mmrazor.models.task_modules.tracer.fx import (
    del_fakequant_after_function, del_fakequant_after_method,
    del_fakequant_after_module, del_fakequant_after_op,
    del_fakequant_before_function, del_fakequant_before_method,
    del_fakequant_before_module, del_fakequant_before_op)
from mmrazor.models.utils import str2class
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import BackendConfigs, QConfigHandler
from .base import BaseQuantizer

if digit_version(torch.__version__) >= digit_version('1.13.0'):
    SUPPORT_QAT_MODULES: Tuple = (
        qat_fused_modules.ConvBn1d, qat_fused_modules.ConvBn2d,
        qat_fused_modules.ConvBn3d, qat_fused_modules.ConvBnReLU1d,
        qat_fused_modules.ConvBnReLU2d, qat_fused_modules.ConvBnReLU3d,
        qat_fused_modules.ConvReLU1d, qat_fused_modules.ConvReLU2d,
        qat_fused_modules.ConvReLU3d, qat_fused_modules.LinearBn1d,
        qat_fused_modules.LinearReLU, qat_modules.Conv1d, qat_modules.Conv2d,
        qat_modules.Conv3d, qat_modules.Linear)

    MERGE_BN_MAPPINGS: Dict = {
        qat_fused_modules.ConvBn1d: qat_modules.Conv1d,
        qat_fused_modules.ConvBn2d: qat_modules.Conv2d,
        qat_fused_modules.ConvBn3d: qat_modules.Conv3d,
        qat_fused_modules.ConvBnReLU1d: qat_fused_modules.ConvReLU1d,
        qat_fused_modules.ConvBnReLU2d: qat_fused_modules.ConvReLU2d,
        qat_fused_modules.ConvBnReLU3d: qat_fused_modules.ConvReLU3d,
        qat_fused_modules.LinearBn1d: qat_modules.Linear
    }

    def fake_quantize_per_channel_affine(g, x, scale, zero_point, ch_axis,
                                         quant_min, quant_max):
        return g.op('mmrazor::FixedPerChannelAffine', x, scale, zero_point,
                    ch_axis, quant_min, quant_max)

    register_custom_op_symbolic('::fake_quantize_per_channel_affine',
                                fake_quantize_per_channel_affine, 11)

    def fake_quantize_per_tensor_affine(g, x, scale, zero_point, quant_min,
                                        quant_max):
        return g.op('mmrazor::FixedPerTensorAffine', x, scale, zero_point,
                    quant_min, quant_max)

    register_custom_op_symbolic('::fake_quantize_per_tensor_affine',
                                fake_quantize_per_tensor_affine, 11)

else:
    SUPPORT_QAT_MODULES = ()
    MERGE_BN_MAPPINGS = {}


@MODELS.register_module()
class TorchNativeQuantizer(BaseQuantizer):
    """Native class for quantizer.

    Args:
        global_qconfig (Union[Dict, Config]): Config for quantization details
            of weight and activation include observer, quantizer, and qscheme.
        no_observer_modules (Optional[List]): Modules don't need observer.
            To fit different backend, we need qconfig to determine the modules
            which don't need observer.
        tracer (Dict): Config for tracer to trace modules for torch fx .

    Raises:
        NotImplementedError: _description_

    Examples:
        >>> global_qconfig = dict(
        ...     w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
        ...     a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
        ...     w_fake_quant=dict(type='mmrazor.FakeQuantize'),
        ...     a_fake_quant=dict(type='mmrazor.FakeQuantize'),
        ...     w_qscheme=dict(
        ...         qdtype='qint8', bit=8, is_symmetry=True,
        ...         is_symmetric_range=True),
        ...     a_qscheme=dict(
        ...         qdtype='quint8', bit=8, is_symmetry=True,
        ...         averaging_constant=0.1),
)
    """

    def __init__(self,
                 global_qconfig: Union[Dict, Config],
                 no_observer_modules: Optional[List] = None,
                 tracer: Dict = dict(type='CustomTracer'),
                 extra_redundant_fakequants: Dict = dict(
                     extra_module_prev_wo_fakequant=tuple(),
                     extra_module_next_wo_fakequant=tuple(),
                     extra_function_prev_wo_fakequant=tuple(),
                     extra_function_next_wo_fakequant=tuple(),
                     extra_method_prev_wo_fakequant=tuple(),
                     extra_method_next_wo_fakequant=tuple(),
                     extra_op_prev_wo_fakequant=tuple(),
                     extra_op_next_wo_fakequant=tuple())):
        super().__init__(tracer)
        self.qconfig = QConfigHandler(global_qconfig)
        if self.qconfig.w_qscheme.is_per_channel:
            w_mode = 'per_channel'
        else:
            w_mode = 'per_tensor'
        if self.qconfig.a_qscheme.is_per_channel:
            a_mode = 'per_channel'
        else:
            a_mode = 'per_tensor'
        assert w_mode in self.support_w_modes
        assert a_mode in self.support_a_modes

        self.qconfig_mapping = self.gen_qconfig_mapping(
            self.qconfig, no_observer_modules)
        self.no_observer_modules = no_observer_modules

        self.backend_config = BackendConfigs[self.backend]
        self.example_inputs = (torch.randn(1, 3, 224, 224), )

        self.extra_redundant_fakequants = extra_redundant_fakequants

    def gen_qconfig_mapping(self, qconfig, no_observer_modules):
        """Convert qconfig in config file to `QConfigMapping`.

        `QConfigMapping` is a custom class for mapping from model ops to
        :class:`torch.ao.quantization.QConfig` s.
        """
        qconfig_mapping = QConfigMapping().set_global(qconfig.convert())

        if no_observer_modules is not None:
            no_observer_modules = str2class(no_observer_modules)
            for mod in no_observer_modules:
                qconfig_mapping.set_object_type(mod, None)

        fixed_qparams_observer_to_qconfig = {}
        for fixed_qparams_op, observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items(
        ):
            if observer in fixed_qparams_observer_to_qconfig:
                fixed_qparams_qconfig = fixed_qparams_observer_to_qconfig[
                    observer]
            else:
                activation = FixedQParamsFakeQuantize.with_args(
                    observer=observer)

                fixed_qparams_qconfig = QConfig(
                    activation=activation, weight=default_weight_fake_quant)
                fixed_qparams_observer_to_qconfig[
                    observer] = fixed_qparams_qconfig
            qconfig_mapping.set_object_type(fixed_qparams_op,
                                            fixed_qparams_qconfig)

        return qconfig_mapping

    @property
    def backend(self):
        """The key of the corresponding backend config."""
        return 'native'

    @property
    def support_w_modes(self):
        """Supported quantization modes for weight about per_tensor or
        per_channel."""
        return ('per_tensor', 'per_channel')

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor')

    def export_onnx(self, model: Union[torch.nn.Module, torch.jit.ScriptModule,
                                       torch.jit.ScriptFunction],
                    args: Union[Tuple[Any, ...],
                                torch.Tensor], output_path: str, **kwargs):
        """Export the onnx model that can be deployed to a native backend."""
        torch.onnx.export(model, args, output_path, **kwargs)

    def prepare(self, model, concrete_args=None):
        """prepare graph to ObservedGraphModule.

        Returns:
            ObservedGraphModule: GraphModules after fuse and observer.

        Notes:
            'graph_module' after '_fuse_fx()' function will fuse conv, BN, ReLU
            into modules in SUPPORT_QAT_MODULES.
            'graph_module' after 'prepare()' function will become observed.

        Notes:
            Keep `is_qat` is True is because in Pytorch when `is_qat` is false,
            the `_fuse_fx()` function only fuse module into `nn.Squential`.
            In mmrazor, we aim to add more ptq algorithm into our pipeline such
            as Adaround, these kind of ptq method have some additional
            fake_quant  operations that we need it to be fused into our
            `SUPPORT_QAT_MODULES` type, which is a tricky way to deal with it.
        """
        self.swap_ff_with_fxff(model)
        traced_graph = self.tracer.trace(model, concrete_args=concrete_args)
        graph_module = build_graphmodule(model, traced_graph)

        # set the training modes of all modules to True to `_fuse_fx` correctly
        # todo: check freezebn
        self.sync_module_training_mode(graph_module, mode=True)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)
        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)
        prepared = self.del_redundant_fakequant(prepared)

        return prepared

    def post_process_for_deploy(self,
                                observed_module: ObservedGraphModule,
                                device: str = 'cpu',
                                update_weight_with_fakequant: bool = False,
                                keep_w_fake_quant: bool = False):
        """weight fake-quant for supported QAT modules.

        Args:
            observed_module (ObservedGraphModule): Modules after fused and
                observed.
            keep_w_fake_quant (bool, optional): Bool to determine whether to
                keep weight fake-quant op, depending on the backend. Defaults
                to False.

        Note:
            `post_process_weight_fakequant()` function is necessary that the
                `SUPPORT_QAT_MODULES` will be convert to normal modules, and
                BN will be really integrated into conv layers.
        """

        def traverse(module):
            for name, child in module.named_children():
                # Trace `SUPPORT_QAT_MODULES` recursively.
                if isinstance(child, SUPPORT_QAT_MODULES):
                    # We add w_fakequant once in case some ptq methods have
                    # specific operations such as Adaround. So we do Quantize
                    # to perform these operations and do dequantize to
                    # introduce quantization loss in advance.
                    weight_fakequant = child.weight_fake_quant

                    # `to_float()` function fuse BN into conv or conv_relu, and
                    # also convert a qat module to a normal module.
                    # source url: https://github.com/pytorch/pytorch/blob/master/torch/nn/intrinsic/qat/modules/conv_fused.py # noqa: E501
                    float_child = child.to_float()

                    if update_weight_with_fakequant:
                        from torch.ao.nn.intrinsic import _FusedModule
                        if issubclass(type(float_child), _FusedModule):
                            float_child[0].weight.data = weight_fakequant(
                                float_child[0].weight.data)
                        else:
                            float_child.weight.data = weight_fakequant(
                                float_child.weight.data)
                    # This is decided by backend type, some backend need
                    # explicitly keep the fake quant structure, others don't.
                    # TODO add deploy doc link
                    if keep_w_fake_quant:
                        # make weight fakequant fixed as the consistent
                        # fakequant, it will help to deploy our model to
                        # various backends.
                        self.qconfig.fixed_w_fakequant()
                        for m in float_child.modules():
                            setattr(m, 'qconfig', self.qconfig.convert())
                        if type(child) in MERGE_BN_MAPPINGS:
                            cls = MERGE_BN_MAPPINGS[type(child)]
                            new_child = cls.from_float(float_child).to(device)
                        else:
                            new_child = type(child).from_float(float_child).to(
                                device)

                        # because weight fakequants and observers are replaced
                        # with base fakequants and base observers, some
                        # initialized args need to be update by running
                        # weight_fake_quant.
                        enable_observer(new_child)
                        new_child.weight_fake_quant(new_child.weight)
                        disable_observer(new_child)
                    else:
                        new_child = float_child.to(device)
                    setattr(module, name, new_child)
                else:
                    traverse(child)

        observed_module.apply(enable_fake_quant)
        observed_module.apply(disable_observer)
        traverse(observed_module)

    def del_redundant_fakequant(self, prepared: GraphModule):
        """delete redundant fakequant op in prepared model.

        Returns:
            prepared (GraphModule): prepared model after delete redundant
                fakequant op.

        Notes:
             We can configure different ways to delete redundant nodes:
                @property
                def module_prev_wo_fakequant(self):
                    return (torch.nn.ReLU6, torch.nn.Identity)
        """
        extra_module_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_module_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_module(
            prepared,
            self.module_prev_wo_fakequant + extra_module_prev_wo_fakequant,
            inplace=True)

        extra_module_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_module_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_module(
            prepared,
            self.module_next_wo_fakequant + extra_module_next_wo_fakequant,
            inplace=True)

        extra_function_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_function_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_function(
            prepared,
            self.function_prev_wo_fakequant + extra_function_prev_wo_fakequant,
            inplace=True)

        extra_function_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_function_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_function(
            prepared,
            self.function_next_wo_fakequant + extra_function_next_wo_fakequant,
            inplace=True)

        extra_method_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_method_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_method(
            prepared,
            self.method_prev_wo_fakequant + extra_method_prev_wo_fakequant,
            inplace=True)

        extra_method_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_method_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_method(
            prepared,
            self.method_next_wo_fakequant + extra_method_next_wo_fakequant,
            inplace=True)

        extra_op_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_op_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_op(
            prepared,
            self.op_prev_wo_fakequant + extra_op_prev_wo_fakequant,
            inplace=True)

        extra_op_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_op_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_op(
            prepared,
            self.op_next_wo_fakequant + extra_op_next_wo_fakequant,
            inplace=True)
        return prepared

    @property
    def module_prev_wo_fakequant(self):
        """Configurate the modules that their previous nodes are redundant
        fakequants."""
        return tuple()

    @property
    def module_next_wo_fakequant(self):
        """Configurate the modules that their next nodes are redundant
        fakequants."""
        return tuple()

    @property
    def function_prev_wo_fakequant(self):
        """Configurate the functions that their previous nodes are redundant
        fakequants."""
        return tuple()

    @property
    def function_next_wo_fakequant(self):
        """Configurate the functions that their next nodes are redundant
        fakequants."""
        return tuple()

    @property
    def method_prev_wo_fakequant(self):
        """Configurate the methods that their previous nodes are redundant
        fakequants."""
        return tuple()

    @property
    def method_next_wo_fakequant(self):
        """Configurate the methods that their next nodes are redundant
        fakequants."""
        return tuple()

    @property
    def op_prev_wo_fakequant(self):
        """Configurate the OPs that their previous nodes are redundant
        fakequants."""
        return tuple()

    @property
    def op_next_wo_fakequant(self):
        """Configurate the OPs that their next nodes are redundant
        fakequants."""
        return tuple()
