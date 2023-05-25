# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.utils import print_log
from .ops import SparseGptConv2d, SparseGptLinear, SparseGptMixIn
from .utils import replace_with_dynamic_ops


def to_static_model(model: nn.Module):
    """Replace dynamicops with torch modules."""
    from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                      load_fix_subnet)
    fix_subnet = export_fix_subnet(model)[0]
    load_fix_subnet(model, fix_subnet)
    return model


class SparseGptCompressor():
    """The compressor with SparseGPT."""

    def __init__(self) -> None:
        self.model: nn.Module = None

    def prepare(self,
                model: nn.Module,
                prune_conv=True,
                prune_linear=True) -> None:
        """Prepare for compressing model."""
        self.model = model
        prune_modules: dict = {}
        if prune_conv:
            prune_modules[nn.Conv2d] = SparseGptConv2d
        if prune_linear:
            prune_modules[nn.Linear] = SparseGptLinear
        replace_with_dynamic_ops(model, prune_modules)

    @classmethod
    def to_static_model(cls, model):
        """Convert replaced op with the original torch model."""
        return to_static_model(model)

    # hessian

    def register_hessian_hooks(self):
        """Register updating hessian hooks for specified ops."""
        for module in self.sparse_ops:
            module.register_hessian_hook()

    def remove_hessian_hooks(self):
        """Remove updating hessian hooks for specified ops."""
        for module in self.sparse_ops:
            module.remove_hessian_hook()

    def init_hessian(self, device=None):
        """Init hessian."""
        for op in self.sparse_ops:
            op.init_hessian(device=device)

    # prune
    def prune(self,
              sparsity,
              prunen=0,
              prunem=0,
              blocksize=128,
              percdamp=.01,
              device=torch.device('cuda')):
        """Apply the compression algorithm to the model."""
        for name, module in self.named_sparse_ops:
            try:
                original_device = next(module.parameters()).device
                module: SparseGptMixIn = module.to(device)
                error = module.prune(
                    sparsity=sparsity,
                    prunen=prunen,
                    prunem=prunem,
                    blocksize=blocksize,
                    percdamp=percdamp,
                )
                print_log(f'prune {name} success \t error = {error}')
                module.to(original_device)
                torch.cuda.empty_cache()
            except Exception as e:
                print_log(f'prune {name} failed as {e}')

    def prune_24(self, device=torch.device('cuda:0')):
        """Apply the compression algorithm to the model with the specified
        setting."""
        self.prune(0.5, prunen=2, prunem=4, device=device)

    # ops

    @property
    def sparse_ops(self):
        """The ops to be applied the algorithm."""
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, SparseGptMixIn):
                yield module

    @property
    def named_sparse_ops(self):
        """The named ops to be applied the algorithm."""
        for name, module in self.model.named_modules():
            if isinstance(module, SparseGptMixIn):
                yield name, module
