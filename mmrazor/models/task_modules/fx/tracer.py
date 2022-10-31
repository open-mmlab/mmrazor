# Copyright (c) OpenMMLab. All rights reserved.
import functools
import inspect
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.fx import GraphModule, Tracer
from torch.fx._compatibility import compatibility
from torch.fx._symbolic_trace import (Graph, _autowrap_check,
                                      _patch_wrapped_functions, _Patcher)
from torch.fx.proxy import Proxy, TraceError

from .graph_module import MMGraphModule

_orig_module_call: Callable = nn.Module.__call__
_orig_module_getattr: Callable = nn.Module.__getattr__


class UntracedMethodRegistry:
    method_dict: Dict[str, nn.Module] = dict()
    tracer = None

    def __init__(self, method):
        self.method = method
        self.instances = dict()
        self.owner = None
        self.name = method.__name__

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        wrapped = self.method_wrapper()
        self.method_dict[name] = dict(mod=self.owner, wrapped=wrapped)

    def __get__(self, instance, owner):
        if instance is None:
            return self.method
        return MethodType(self.method, instance)

    def method_wrapper(self):

        @functools.wraps(self.method)
        def wrapped_method(mod, *args, **kwargs):

            def method(*args, **kwargs):
                return self.method(mod, *args, **kwargs)

            return self.tracer.call_method(mod, self.name, method, args,
                                           kwargs)

        return wrapped_method


def get_class_that_defined_method(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    if inspect.isfunction(meth):
        return getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0], None)
    return None


def register_skipped_method(method):
    if not isinstance(method, list):
        method = [method]
    for met in method:
        assert isinstance(
            met, FunctionType
        ), f'Expecting {met} is `FunctionType`, but got `{type(met)}`'
        _registry = UntracedMethodRegistry(met)
        UntracedMethodRegistry.method_dict[met.__name__] = dict(
            mod=get_class_that_defined_method(met),
            wrapped=_registry.method_wrapper())


def custom_symbolic_trace(root, concrete_args=None):
    tracer = CustomTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(
        root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


def _parse_method_string(method_string):
    import importlib
    pkg, string = method_string.split('.', 1)
    mod = importlib.import_module(pkg)
    while '.' in string:
        module_name, string = string.split('.', 1)
        mod = getattr(mod, module_name)

    mod = getattr(mod, string)
    return mod


@compatibility(is_backward_compatible=True)
def custom_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    customed_skipped_method: Optional[Dict[str, Any]] = None,
) -> MMGraphModule:

    tracer = CustomTracer(customed_skipped_method=customed_skipped_method)
    root.mode = 'loss'  # type: ignore
    graph_loss = tracer.trace(root, concrete_args)
    root.mode = 'predict'  # type: ignore
    graph_predict = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__
        if isinstance(root, torch.nn.Module) else root.__name__)
    graphs = dict(loss=graph_loss, predict=graph_predict)
    return MMGraphModule(tracer.root, graphs, name)


class CustomTracer(Tracer):

    def __init__(self, *args, customed_skipped_method=None, **kwargs):
        super(CustomTracer, self).__init__()
        UntracedMethodRegistry.tracer = self
        self.customed_skipped_method = customed_skipped_method
        if self.customed_skipped_method:
            self.register_skipped_module()

    def register_skipped_module(self):
        if not isinstance(self.customed_skipped_method, list):
            self.customed_skipped_method = [self.customed_skipped_method]
        method_list = []
        for method_str in self.customed_skipped_method:
            method = _parse_method_string(method_str)
            method_list.append(method)
        register_skipped_method(method_list)

    def call_method(self, m: torch.nn.Module, name, method, args, kwargs):
        """Method that specifies the behavior of this ``Tracer`` when it
        encounters a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf
        module via ``is_leaf_module``. If it is, emit a ``call_module`` node
        referring to ``m`` in the ``Graph``. Otherwise, call the ``Module``
        normally, tracing through the operations in its ``forward`` function.
        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.
        Args:
            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be
                invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite
        Return:
            The return value from the Module call. In the case that a
            ``call_module`` node was emitted, this is a ``Proxy`` value.
            Otherwise, it is whatever value was returned from the ``Module``
            invocation.
        """
        # module_qualified_name = self.path_of_module(m)
        if not self.is_skipped_method(m):
            return method(*args, **kwargs)
        args = list(args)
        args.insert(0, m)
        args = tuple(args)
        return self.create_proxy('call_method', name, args, kwargs)

    def trace(self, root, concrete_args=None):
        if isinstance(root, torch.nn.Module):
            self.root = root
            fn = type(root).forward
            self.submodule_paths = {
                mod: name
                for name, mod in root.named_modules()
            }
        else:
            self.root = torch.nn.Module()
            fn = root

        tracer_cls: Optional[Type['Tracer']] = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)

        # When we encounter a Tensor value that's not a parameter, we look if
        # it is some other attribute on the model. Construct a dict mapping
        # Tensor values to the qualified name here for efficiency. This is
        # used downstream in create_arg
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

        # Method dispatch on parameters is not recorded unless
        # it's directly used. Thus, we need to insert a proxy
        # when __getattr__ requests a parameter.
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

            for name, value in UntracedMethodRegistry.method_dict.items():
                wrapped = value['wrapped']
                patcher.patch_method(
                    value['mod'], name, wrapped, deduplicate=False)

            _patch_wrapped_functions(patcher)
            _autowrap_check(patcher, fn_globals, self._autowrap_function_ids)
            for module in self._autowrap_search:
                _autowrap_check(patcher, module.__dict__,
                                self._autowrap_function_ids)
            self.create_node(
                'output',
                'output', (self.create_arg(fn(*args)), ), {},
                type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None

        return self.graph

    def is_skipped_method(self, m):
        mods = tuple(value['mod']
                     for value in UntracedMethodRegistry.method_dict.values())
        custom = isinstance(m, mods)
        return custom

    def iter(self, obj: 'Proxy') -> Iterator:
        """Called when a proxy object is being iterated over, such as when used
        in control flow.

        Normally we don't know what to do because we don't know the value of
        the proxy, but a custom tracer can attach more information to the graph
        node using create_node and can choose to return an iterator.
        """
        raise TraceError('Proxy object cannot be iterated. This can be '
                         'attempted when the Proxy is used in a loop or'
                         ' as a *args or **kwargs function argument. '
                         'See the torch.fx docs on pytorch.org for a '
                         'more detailed explanation of what types of '
                         'control flow can be traced, and check out the'
                         ' Proxy docstring for help troubleshooting '
                         'Proxy iteration errors')
