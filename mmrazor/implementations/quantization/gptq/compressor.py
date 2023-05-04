# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.utils import print_log
from .ops import GPTQLinear, GPTQConv2d, GPTQMixIn
from .utils import replace_with_dynamic_ops
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
                quant_linear=True) -> None:
        self.model = model
        quant_modules: dict = {}
        if quant_conv:
            quant_modules[nn.Conv2d] = GPTQConv2d
        if quant_linear:
            quant_modules[nn.Linear] = GPTQLinear
        replace_with_dynamic_ops(model, quant_modules)

    @classmethod
    def to_static_model(cls, model):
        return to_static_model(model)

    # hessian

    def start_init_hessian(self):
        for module in self.sparse_ops:
            module.start_init_hessian()

    def end_init_hessian(self):
        for module in self.sparse_ops:
            module.end_init_hessian()

    def keep_hessian_in_float(self):
        for op in self.sparse_ops:
            op.keep_hessian_in_float()

    # quant
    def quant(self,
              quantizer,
              blocksize=128,
              percdamp=0.01,
              groupsize=-1,
              actorder=False,
              device=torch.device('cuda')):
        for name, module in self.named_quant_ops:
            try:
                original_device = next(module.parameters()).device
                module: GPTQMixIn = module.to(device)
                error = module.quant(
                    quantizer=quantizer,
                    blocksize=blocksize,
                    percdamp=percdamp,
                    groupsize=groupsize,
                    actorder=actorder
                )
                print_log(f'quant {name} success \t error = {error}')
                module.to(original_device)
                torch.cuda.empty_cache()
            except Exception as e:
                print_log(f'quant {name} failed as {e}')
    
    def quant_default_setting(self, device=torch.device('cuda:0')):
        quantizer = Quantizer()
        self.quant(quantizer=quantizer)

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
