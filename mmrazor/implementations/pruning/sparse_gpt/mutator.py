# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmrazor.utils import print_log
from .ops import SparseGptConv2d, SparseGptLinear, SparseGptMixIn
from .utils import replace_with_dynamic_ops


class SparseGptMutator():

    def __init__(self, sparse_model: nn.Module) -> None:
        self.model = sparse_model

    def start_init_hessian(self):
        for module in self.sparse_ops:
            module.start_init_hessian()

    def end_init_hessian(self):
        for module in self.sparse_ops:
            module.end_init_hessian()

    def prune_24(self):
        for name, module in self.named_sparse_ops:
            try:
                error = module.prune_24()
                print_log(f'prune {name} success \t error = {error}')
            except Exception as e:
                print_log(f'prune {name} failed as {e}')

    @property
    def sparse_ops(self):
        for module in self.model.modules():
            if isinstance(module, SparseGptMixIn):
                yield module

    @property
    def named_sparse_ops(self):
        for name, module in self.model.named_modules():
            if isinstance(module, SparseGptMixIn):
                yield name, module

    @classmethod
    def init_from_a_model(cls, model: nn.Module):
        replace_with_dynamic_ops(model, {
            nn.Linear: SparseGptLinear,
            nn.Conv2d: SparseGptConv2d
        })
        mutator = cls(model)
        return mutator
