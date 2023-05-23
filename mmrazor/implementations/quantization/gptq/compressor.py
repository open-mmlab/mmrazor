# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Type

import torch
import torch.nn as nn

from mmrazor.utils import print_log
from .ops import GPTQConv2d, GPTQLinear, GPTQMixIn, TritonGPTQLinear
from .quantizer import Quantizer


def replace_with_dynamic_ops(model: nn.Module,
                             dynamicop_map: Dict[Type[nn.Module], Type[Any]],
                             skipped_layers=[],
                             a_qconfig=None,
                             **kwargs):
    """Replace torch modules with dynamic-ops."""

    def replace_op(model: nn.Module, name: str, module: nn.Module):
        names = name.split('.')
        for sub_name in names[:-1]:
            model = getattr(model, sub_name)

        setattr(model, names[-1], module)

    for name, module in model.named_modules():
        if type(module) in dynamicop_map and name not in skipped_layers:
            if isinstance(module, nn.Linear):
                if a_qconfig:
                    a_fakequant = Quantizer()
                    a_fakequant.configure(**a_qconfig)
                    kwargs.update({'a_fakequant': a_fakequant})
                new_module = dynamicop_map[type(module)].convert_from(
                    module, **kwargs)
            else:
                new_module = dynamicop_map[type(module)].convert_from(module)
            replace_op(model, name, new_module)


def to_static_model(model: nn.Module):
    """Replace dynamicops with torch modules."""
    from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                      load_fix_subnet)
    fix_subnet = export_fix_subnet(model)[0]
    load_fix_subnet(model, fix_subnet)
    return model


class GPTQCompressor():
    """The compressor with GPTQ."""

    def __init__(self) -> None:
        self.model: nn.Module = None

    def prepare(self,
                model: nn.Module,
                quant_conv=True,
                quant_linear=True,
                use_triton_ops=True,
                skipped_layers=[],
                a_qconfig=None,
                **kwargs) -> None:
        """Prepare for compressing model."""
        self.model = model
        quant_modules: dict = {}
        if quant_conv:
            quant_modules[nn.Conv2d] = GPTQConv2d
        if quant_linear:
            gptq_linear = TritonGPTQLinear if use_triton_ops else GPTQLinear
            quant_modules[nn.Linear] = gptq_linear
        replace_with_dynamic_ops(model, quant_modules, skipped_layers,
                                 a_qconfig, **kwargs)

    @classmethod
    def to_static_model(cls, model):
        """Convert replaced op with the original torch model."""
        return to_static_model(model)

    # hessian

    def register_hessian_hooks(self):
        """Register updating hessian hooks for specified ops."""
        for module in self.quant_ops:
            module.register_hessian_hook()

    def remove_hessian_hooks(self):
        """Remove updating hessian hooks for specified ops."""
        for module in self.quant_ops:
            module.remove_hessian_hook()

    def init_hessian(self, device=None):
        """Init hessian."""
        for op in self.quant_ops:
            op.init_hessian(device=device)

    # quant
    def quant(self,
              blocksize=128,
              percdamp=0.01,
              groupsize=-1,
              actorder=False,
              device=torch.device('cuda:0'),
              **qconfig):
        """Apply the compression algorithm to the model."""
        for name, module in self.named_quant_ops:
            try:
                original_device = next(module.parameters()).device
                module: GPTQMixIn = module.to(device)
                quantizer = Quantizer()
                quantizer.configure(**qconfig)
                # print_log(f'quant {name}...')
                error = module.quant(
                    quantizer=quantizer,
                    blocksize=blocksize,
                    percdamp=percdamp,
                    groupsize=groupsize,
                    actorder=actorder)
                print_log(f'quant {name} success \t error = {error}')
                module.to(original_device)
                module.free()
            except Exception as e:
                print_log(f'quant {name} failed as {e}')

    def quant_with_default_qconfig(self, groupsize=128, device='cpu'):
        """Apply the compression algorithm to the model with the specified
        setting."""
        qconfig = dict(bits=4, perchannel=True, sym=False)
        self.quant(
            groupsize=groupsize, actorder=True, device=device, **qconfig)

    # ops

    @property
    def quant_ops(self):
        """The ops to be applied the algorithm."""
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, GPTQMixIn):
                yield module

    @property
    def named_quant_ops(self):
        """The named ops to be applied the algorithm."""
        for name, module in self.model.named_modules():
            if isinstance(module, GPTQMixIn):
                yield name, module
