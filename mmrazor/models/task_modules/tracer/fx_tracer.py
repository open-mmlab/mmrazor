# Copyright (c) OpenMMLab. All rights reserved.
"""This module define FxTracer and related classes."""

import functools
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.fx as fx
import torch.nn as nn
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.fx._symbolic_trace import (Tracer, _autowrap_check,
                                      _orig_module_call, _orig_module_getattr,
                                      _patch_wrapped_functions, _Patcher)
from torch.fx.graph import Graph
from torch.fx.node import Argument
from torch.fx.proxy import Proxy


class CostumFxTracer(Tracer):
    """CostumFxTracer allow user to indicate leaf module."""

    def __init__(self,
                 is_extra_leaf_module: Callable[[nn.Module, str], bool] = None,
                 warp_method={},
                 concrete_args={}) -> None:
        """
        Args:
            is_extra_leaf_module: Callable[[nn.Module, str], bool]: a function
            to determine if a module is a leaf module except torch pre-defined
            modules.
        """
        super().__init__(
            param_shapes_constant=True,
            autowrap_functions=[torch.arange],
        )
        self.extra_is_leaf_module = is_extra_leaf_module
        self.concrete_args = concrete_args
        from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
        from mmdet.models.dense_heads.rpn_head import RPNHead
        from mmdet.models.roi_heads import StandardRoIHead
        self.warp_method = {
            RPNHead: RPNHead.predict_by_feat,
            BaseDenseHead: BaseDenseHead.predict_by_feat,
            StandardRoIHead: StandardRoIHead.forward,
        }
        self.warp_fn = {
            torch: torch.arange,
        }

    def is_leaf_module(self, m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        """Bool: determine if a module is a leaf module"""
        is_torch_module = super().is_leaf_module(m, module_qualified_name)
        if self.extra_is_leaf_module is None:
            is_extra = False
        else:
            is_extra = self.extra_is_leaf_module(m, module_qualified_name)
        return is_torch_module or is_extra

    def trace(self, root) -> fx.graph.Graph:
        return self._trace(root, self.concrete_args)

    def _trace(self,
               root: Union[torch.nn.Module, Callable[..., Any]],
               concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        if isinstance(root, torch.nn.Module):
            self.root = root

            assert hasattr(type(root), self.traced_func_name), (
                f"traced_func_name={self.traced_func_name} doesn't exist in"
                ' {type(root).__name__}')

            fn = getattr(type(root), self.traced_func_name)
            self.submodule_paths = {
                mod: name
                for name, mod in root.named_modules()
            }
        else:
            self.root = torch.nn.Module()
            fn = root

        tracer_cls: Optional[Type['Tracer']] = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look
        # if it
        # is some other attribute on the model. Construct a dict mapping
        # Tensor
        # values to the qualified name here for efficiency. This is used
        # downstream
        # in create_arg
        self.tensor_attrs: Dict[Union[torch.Tensor, ScriptObject], str] = {}

        def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
            for k, v in m.__dict__.items():
                if isinstance(v, (torch.Tensor, ScriptObject)):
                    self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
            for k, v in m.named_children():
                collect_tensor_attrs(v, prefix_atoms + [k])

        collect_tensor_attrs(self.root, [])

        assert isinstance(fn, FunctionType)

        fn_globals = fn.__globals__  # run before it gets patched
        fn, args = self.create_args_for_root(fn,
                                             isinstance(root, torch.nn.Module),
                                             concrete_args)

        parameter_proxy_cache: Dict[str, Proxy] = {
        }  # Reduce number of get_attr calls

        # Method dispatch on parameters is not recorded unless it's directly
        # used.
        # Thus, we need to insert a proxy when __getattr__ requests a
        # parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            return self._module_getattr(attr, attr_val, parameter_proxy_cache)

        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):

            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(
                patcher,
                getattr(getattr(mod, 'forward', mod), '__globals__', {}),
                self._autowrap_function_ids)
            return self.call_module(mod, forward, args, kwargs)

        with _Patcher() as patcher:
            # allow duplicate patches to support the case of nested calls
            patcher.patch_method(
                torch.nn.Module,
                '__getattr__',
                module_getattr_wrapper,
                deduplicate=False)
            patcher.patch_method(
                torch.nn.Module,
                '__call__',
                module_call_wrapper,
                deduplicate=False)
            for obj, mth in self.warp_method.items():
                patcher.patch_method(
                    obj,
                    mth.__name__,
                    self.warp_a_method(obj, mth),
                    deduplicate=False)
            for obj, mth in self.warp_fn.items():
                patcher.patch_method(
                    obj,
                    mth.__name__,
                    self.warp_a_function(obj, mth),
                    deduplicate=False)
            _patch_wrapped_functions(patcher)
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__,
                                self._autowrap_function_ids)
            self.create_node(
                'output',
                'output', (self.create_arg(fn(*args)), ), {},
                type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None  # type: ignore

        return self.graph

    def call_method(self, origin_fn, name, args: tuple, kwargs):
        args = args[1:]
        return self.create_proxy('call_function', origin_fn, args, kwargs,
                                 name)

    def call_function(self, origin_fn, name, args, kwargs):
        return self.create_proxy('call_function', origin_fn, args, kwargs,
                                 name)

    def warp_a_method(self, obj, origin_fn):

        @functools.wraps(origin_fn)
        def fn_wrapper(*args, **kwargs):
            return self.call_method(origin_fn, origin_fn.__name__, args,
                                    kwargs)

        return fn_wrapper

    def warp_a_function(self, obj, origin_fn):

        @functools.wraps(origin_fn)
        def fn_wrapper(*args, **kwargs):
            return self.call_function(origin_fn, origin_fn.__name__, args,
                                      kwargs)

        return fn_wrapper

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        module_qualified_name = self.path_of_module(m)
        try:
            proxy = super().call_module(m, forward, args, kwargs)
            return proxy
        except Exception as e:
            module_qualified_name = self.path_of_module(m)
            from mmengine import MMLogger
            MMLogger.get_current_instance().warning(
                f'{module_qualified_name}({type(m)}) encounter error when'
                ' tracing. '
                f'It will be treated as a leaf module.\n {e}')
            return self.create_proxy('call_module', module_qualified_name,
                                     args, kwargs)

    def create_arg(self, a: Any) -> 'Argument':
        try:
            arg = super().create_arg(a)
            return arg
        except Exception:
            return a
