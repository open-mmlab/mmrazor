# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.implementations.pruning.sparse_gpt import replace_with_dynamic_ops
from mmrazor.utils import print_log
from .ops import GPTQConv2d, GPTQLinear, GPTQMixIn, TritonGPTQLinear
from .quantizer import Quantizer


def to_static_model(model: nn.Module):
    from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                      load_fix_subnet)
    fix_subnet = export_fix_subnet(model)[0]
    load_fix_subnet(model, fix_subnet)
    return model


class GPTQCompressor():

    # init

    def __init__(self) -> None:
        self.model: nn.Module = None

    def prepare(self,
                model: nn.Module,
                quant_conv=True,
                quant_linear=True,
                use_triton_ops=True,
                skipped_layers=[],
                **kwargs) -> None:
        self.model = model
        quant_modules: dict = {}
        if quant_conv:
            quant_modules[nn.Conv2d] = GPTQConv2d
        if quant_linear:
            gptq_linear = TritonGPTQLinear if use_triton_ops else GPTQLinear
            quant_modules[nn.Linear] = gptq_linear
        replace_with_dynamic_ops(model, quant_modules, skipped_layers,
                                 **kwargs)

    @classmethod
    def to_static_model(cls, model):
        return to_static_model(model)

    # hessian

    def register_hessian_hook(self):
        for module in self.quant_ops:
            module.register_hessian_hook()

    def remove_hessian_hook(self):
        for module in self.quant_ops:
            module.remove_hessian_hook()

    def init_hessian(self, device=None):
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
        for name, module in self.named_quant_ops:
            try:
                original_device = next(module.parameters()).device
                module: GPTQMixIn = module.to(device)
                quantizer = Quantizer()
                quantizer.configure(**qconfig)
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

    def quant_with_default_qconfig(self, device='cpu'):
        qconfig = dict(bits=4, perchannel=True, sym=False)
        self.quant(groupsize=128, actorder=True, device=device, **qconfig)

    # ops

    @property
    def quant_ops(self):
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, GPTQMixIn):
                yield module

    @property
    def named_quant_ops(self):
        for name, module in self.model.named_modules():
            if isinstance(module, GPTQMixIn):
                yield name, module
