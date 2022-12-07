# Copyright (c) OpenMMLab. All rights reserved.
"""This module define FxTracer and related classes."""
# flake8: noqa
import functools
import inspect
import sys
import types
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from mmengine import MMLogger
from torch._C import ScriptObject  # type: ignore[attr-defined]

from mmrazor.utils import get_placeholder

try:
    from torch.fx._symbolic_trace import (Tracer, _find_proxy,
                                          _orig_module_call,
                                          _orig_module_getattr,
                                          _patch_wrapped_functions, _Patcher)
    from torch.fx.graph import Graph
    from torch.fx.node import Argument
    from torch.fx.proxy import Proxy
except ImportError:
    Tracer = get_placeholder('torch>=1.12')
    _find_proxy = get_placeholder('torch>=1.12')
    _orig_module_call = get_placeholder('torch>=1.12')
    _orig_module_getattr = get_placeholder('torch>=1.12')
    _patch_wrapped_functions = get_placeholder('torch>=1.12')
    _Patcher = get_placeholder('torch>=1.12')
    Graph = get_placeholder('torch>=1.12')
    Argument = get_placeholder('torch>=1.12')
    Proxy = get_placeholder('torch>=1.12')

from mmrazor import digit_version

sys.setrecursionlimit(int(pow(2, 20)))

logger = MMLogger.get_current_instance()


def _autowrap_check(patcher: _Patcher, frame_dict: Dict[str, Any],
                    function_ids: Set[int]):
    auto_wrapper = AutoWrapper(patcher)
    auto_wrapper.wrap(None, '', frame_dict)


def auto_wrap(patcher, owner):
    auto_wrapper = AutoWrapper(patcher)
    auto_wrapper.wrap(None, '', owner)


class AutoWrapper:

    def __init__(self, patcher) -> None:
        self.patcher: _Patcher = patcher

    # wrap

    def wrap(self, owner, name, val):

        def is_method(val):
            return (inspect.ismethod(val) or inspect.isfunction(val)
                    or isinstance(val, types.BuiltinFunctionType)
                    or isinstance(val, staticmethod)
                    or isinstance(val, classmethod))

        if owner is None and isinstance(val, dict):
            self.wrap_frame(owner, name, val)
        else:
            # class
            if inspect.isclass(val):
                self.wrap_class(owner, name, val)
            # method
            elif inspect.isclass(owner) and is_method(val):
                self.wrap_method(owner, name, val)
            # function
            elif inspect.isfunction(val) or isinstance(
                    val, types.BuiltinFunctionType):
                self.wrap_function(owner, name, val)
            # package
            elif isinstance(val, ModuleType):
                self.wrap_module(owner, name, val)
            # instance
            elif isinstance(val, object):
                self.wrap_class(None, '', type(val))
            # else
            else:
                logger.debug(f'unsupported type to wrap: {name}/{type(val)}')

    def wrap_frame(self, owner, name: str, val: dict):
        assert isinstance(val, dict)

        if self.patcher.visit_once(val):
            frame_name = val['__name__'] if '__name__' in val else ''
            logger.debug(f'wrap a frame {frame_name}')
            for key in val:
                self.wrap(val, key, val[key])

    def wrap_module(self, owner, name, val):
        if self.visit_once(val):
            if val in [torch]:
                logger.debug(f'wrap a module {owner[name]}')
                self.wrap(None, '', val.__dict__)

    def wrap_class(self, owner, name, val):
        assert inspect.isclass(val)
        if issubclass(val, nn.Module):
            if self.visit_once(val):
                logger.debug(f'wrap a class {val}')
                for key in val.__dict__:
                    key: str
                    if not (key.startswith('__')):
                        self.wrap(val, key, val.__dict__[key])

    def wrap_function(self, owner, name, val):
        if self.visit_once(val):
            self.patcher.patch(owner, name, self.func_wapper(val))
            logger.debug(f'wrap a function {name}')

    def wrap_method(self, owner, name, val):
        assert inspect.isclass(owner)
        if self.visit_once(val):
            try:
                if isinstance(val, staticmethod):
                    pass
                    logger.debug(f'wrap a staticmethod {name} (unimplement)')
                elif isinstance(val, classmethod):
                    pass
                    logger.debug(f'wrap a classmethod {name} (unimplement)')
                else:
                    self.patcher.patch_method(owner, name,
                                              self.method_wrapper(val))
                    logger.debug(f'wrap an instance method {name}')
            except Exception:
                self.patcher.patches_made.pop()

    # wrapper
    def func_wapper(self, orig_fn):

        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            """Given an closed-over ``orig_function`` to invoke, search the
            args and kwargs for a Proxy object.

            If there is one, emit a ``call_function`` node to preserve the call
            to this leaf function directly. Otherwise, just return the results
            of this function call, as this function is not being traced.
            """
            _autowrap_check(self.patcher, getattr(orig_fn, '__globals__', {}),
                            set())
            try:
                end = orig_fn(*args, **kwargs)
                return end
            except Exception:
                logger.debug(f'auto wrap {orig_fn}')
                proxy = _find_proxy(args, kwargs)
                if proxy is not None:
                    return_proxy = proxy.tracer.create_proxy(
                        'call_function', orig_fn, args, kwargs)
                    return_proxy.node.meta['is_wrapped'] = True
                    return return_proxy
                else:
                    return orig_fn(*args, **kwargs)

        return wrapped

    def method_wrapper(self, orig_fn):

        @functools.wraps(orig_fn)
        def wrapped(*args, **kwargs):
            """Given an closed-over ``orig_function`` to invoke, search the
            args and kwargs for a Proxy object.

            If there is one, emit a ``call_function`` node to preserve the call
            to this leaf function directly. Otherwise, just return the results
            of this function call, as this function is not being traced.
            """
            _autowrap_check(self.patcher, getattr(orig_fn, '__globals__', {}),
                            set())
            # logger.debug(f'call method {orig_fn}')
            try:
                end = orig_fn(*args, **kwargs)
                return end
            except Exception:
                logger.debug(f'auto wrap {orig_fn}')
                proxy: Proxy = _find_proxy(args, kwargs)
                if proxy is not None:
                    return_proxy = proxy.tracer.create_proxy(
                        'call_method', orig_fn.__name__, args, kwargs)
                    return_proxy.node.meta['is_wrapped'] = True
                    return return_proxy
                else:
                    return orig_fn(*args, **kwargs)

        return wrapped

    # others
    def visit_once(self, obj):
        return self.patcher.visit_once(obj)

    def is_visited(self, obj):
        id_ = id(obj)
        return id_ in self.patcher.visited


class FxTracer(Tracer):

    def trace(self,
              root: Union[torch.nn.Module, Callable[..., Any]],
              concrete_args: Optional[Dict[str, Any]] = None) -> Graph:
        """Please refer to torch.fx._symbolic_trace.Tracer."""
        if isinstance(root, torch.nn.Module):
            self.root = root

            assert hasattr(type(root), self.traced_func_name), (
                f"traced_func_name={self.traced_func_name} doesn't exist in {type(root).__name__}"  # noqa
            )  # noqa

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

        # When we encounter a Tensor value that's not a parameter, we look if it
        # is some other attribute on the model. Construct a dict mapping Tensor
        # values to the qualified name here for efficiency. This is used downstream
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

        # Method dispatch on parameters is not recorded unless it's directly used.
        # Thus, we need to insert a proxy when __getattr__ requests a parameter.
        @functools.wraps(_orig_module_getattr)
        def module_getattr_wrapper(mod, attr):
            attr_val = _orig_module_getattr(mod, attr)
            ########################################################################
            if digit_version(torch.__version__) >= digit_version('1.13.0'):
                return self.getattr(attr, attr_val, parameter_proxy_cache)
            else:
                return self._module_getattr(attr, attr_val,
                                            parameter_proxy_cache)
            ########################################################################
        @functools.wraps(_orig_module_call)
        def module_call_wrapper(mod, *args, **kwargs):

            def forward(*args, **kwargs):
                return _orig_module_call(mod, *args, **kwargs)

            _autowrap_check(
                patcher,
                getattr(getattr(mod, 'forward', mod), '__globals__', {}),
                self._autowrap_function_ids)
            ########################################################################
            auto_wrap(patcher, mod)
            ########################################################################

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
            _patch_wrapped_functions(patcher)
            ########################################################################
            patcher.visit_once(globals())
            auto_wrap(patcher, self.root)
            ########################################################################
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__,
                                self._autowrap_function_ids)
            self.create_node(
                'output',
                'output', (self.create_arg(fn(*args)), ), {},
                type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None  # type:ignore

        return self.graph

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:

        try:
            return super().call_module(m, forward, args, kwargs)
        except Exception:
            module_qualified_name = self.path_of_module(m)
            return self.create_proxy('call_module', module_qualified_name,
                                     args, kwargs)

    def create_arg(self, a: Any) -> 'Argument':
        try:
            arg = super().create_arg(a)
            return arg
        except Exception:
            return a


class MMFxTracer(FxTracer):

    def __init__(
            self,
            autowrap_modules: Tuple = (),
            autowrap_functions: Tuple[Callable, ...] = (),
            param_shapes_constant: bool = False,
            leaf_module: Tuple = (),
    ) -> None:
        super().__init__(autowrap_modules, autowrap_functions,
                         param_shapes_constant)

        self.leaf_module = leaf_module

    def is_leaf_module(self, m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        is_torch_module = super().is_leaf_module(m, module_qualified_name)

        is_leaf = False
        for module_type in self.leaf_module:
            if isinstance(m, module_type):
                is_leaf = True
                break

        return is_leaf or is_torch_module
