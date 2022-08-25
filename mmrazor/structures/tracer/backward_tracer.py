# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
from collections import OrderedDict

from mmengine import ConfigDict
from torch.nn import Conv2d, Linear
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _NormBase

from mmrazor.registry import TASK_UTILS
from .parsers import DEFAULT_BACKWARD_TRACER
from .path import Path, PathList

SUPPORT_MODULES = (Conv2d, Linear, _NormBase, GroupNorm)


@TASK_UTILS.register_module()
class BackwardTracer:
    """A topology tracer via backward.

    Args:
        loss_calculator (dict or Callable): Calculate the pseudo loss to trace
            the topology of a model.
    """

    def __init__(self, loss_calculator):
        if isinstance(loss_calculator, (dict, ConfigDict)):
            loss_calculator = TASK_UTILS.build(loss_calculator)

        assert callable(
            loss_calculator
        ), 'loss_calculator should be a dict, ConfigDict or ' \
           'callable object'
        self.loss_calculator = loss_calculator

    @property
    def backward_parser(self):
        """The mapping from the type of a backward op to the corresponding
        parser."""
        return DEFAULT_BACKWARD_TRACER

    def backward_trace(self, grad_fn, module2name, param2module, cur_path,
                       result_paths, visited, shared_module):
        """Trace the topology of all the ``NON_PASS_MODULE``."""
        grad_fn = grad_fn[0] if isinstance(grad_fn, (list, tuple)) else grad_fn

        if grad_fn is not None:
            name = type(grad_fn).__name__
            # In pytorch graph, there may be an additional '0' or '1'
            # (e.g. ThnnConv2DBackward0) after a backward op. Delete the
            # digit numbers to build the corresponding parser.
            name = re.sub(r'[0-1]+', '', name)
            parse_module = self.backward_parser.get(name)

            if parse_module is not None:
                parse_module(self, grad_fn, module2name, param2module,
                             cur_path, result_paths, visited, shared_module)
            else:
                # If the op is AccumulateGrad, parents is (),
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        self.backward_trace(parent, module2name, param2module,
                                            cur_path, result_paths, visited,
                                            shared_module)
        else:
            result_paths.append(copy.deepcopy(cur_path))

    def _trace_shared_module_hook(self, module, inputs, outputs):
        """Trace shared modules. Modules such as the detection head in
        RetinaNet which are visited more than once during :func:`forward` are
        shared modules.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        module._cnt += 1

    def _build_mappings(self, model):
        """Build the mappings which are used during tracing."""

        module2name = OrderedDict()
        # build a mapping from the identity of a module's params
        # to this module
        param2module = OrderedDict()
        # record the visited module name during trace path
        visited = dict()

        def traverse(module, prefix=''):
            for name, child in module.named_children():
                full_name = f'{prefix}.{name}' if prefix else name

                if isinstance(child, SUPPORT_MODULES):
                    module2name[child] = full_name
                    for param in child.parameters():
                        param2module[id(param)] = child
                    visited[full_name] = False
                else:
                    traverse(child, full_name)

        traverse(model)

        return module2name, param2module, visited

    def _register_share_module_hook(self, model):
        """Record shared modules which will be visited more than once during
        forward such as shared detection head in RetinaNet.

        If a module is not a shared module and it has been visited during
        forward, its parent modules must have been traced already. However, a
        shared module will be visited more than once during forward, so it is
        still need to be traced even if it has been visited.
        """
        self._shared_module_hook_handles = list()
        for module in model.modules():
            if hasattr(module, 'weight'):
                # trace shared modules
                module._cnt = 0
                # the handle is only to remove the corresponding hook later
                handle = module.register_forward_hook(
                    self._trace_shared_module_hook)
                self._shared_module_hook_handles.append(handle)

    def _remove_share_module_hook(self, model):
        """`_trace_shared_module_hook` and `_cnt` are only used to trace the
        shared modules in a model and need to be remove later."""
        for module in model.modules():
            if hasattr(module, 'weight'):
                del module._cnt

        for handle in self._shared_module_hook_handles:
            handle.remove()

        del self._shared_module_hook_handles

    def _set_all_requires_grad(self, model):
        """Set `requires_grad` of a parameter to True to trace the whole
        architecture topology."""
        self._param_requires_grad = dict()
        for param in model.parameters():
            self._param_requires_grad[id(param)] = param.requires_grad
            param.requires_grad = True

    def _restore_requires_grad(self, model):
        """We set requires_grad to True to trace the whole architecture
        topology.

        So it should be reset after that.
        """
        for param in model.parameters():
            param.requires_grad = self._param_requires_grad[id(param)]
        del self._param_requires_grad

    @staticmethod
    def _find_share_modules(model):
        """Find shared modules which will be visited more than once during
        forward such as shared detection head in RetinaNet."""
        share_modules = list()
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                if module._cnt > 1:
                    share_modules.append(name)

        return share_modules

    @staticmethod
    def _reset_norm_running_stats(model):
        """As we calculate the pseudo loss during tracing, we need to reset
        states of parameters."""
        for module in model.modules():
            if isinstance(module, _NormBase):
                module.reset_parameters()

    def trace(self, model):
        """Trace trace the architecture topology of the input model."""
        module2name, param2module, visited = self._build_mappings(model)

        # Set requires_grad to True. If the `requires_grad` of a module's
        # weight is False, we can not trace this module by parsing backward.
        self._set_all_requires_grad(model)

        self._register_share_module_hook(model)

        pseudo_loss = self.loss_calculator(model)

        share_modules = self._find_share_modules(model)

        self._remove_share_module_hook(model)
        self._restore_requires_grad(model)

        module_path_list = PathList()

        self.backward_trace(pseudo_loss.grad_fn, module2name, param2module,
                            Path(), module_path_list, visited, share_modules)

        self._reset_norm_running_stats(model)

        return module_path_list
