# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import FunctionType, MethodType
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from mmengine.utils import import_modules_from_strings
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch.ao.quantization.quantize_fx import QuantizationTracer
from torch.fx import GraphModule, Tracer
from torch.fx._symbolic_trace import (Graph, _autowrap_check,
                                      _patch_wrapped_functions, _Patcher)
from torch.fx.proxy import Proxy

_orig_module_call: Callable = torch.nn.Module.__call__
_orig_module_getattr: Callable = torch.nn.Module.__getattr__
# _orig_module_forward_train: Callable = models.BaseDenseHead.forward_train


class UntracedMethodRegistry:
    method_dict: Dict = dict()
    tracer = None

    def __init__(self, method):
        self.method = method
        self.instances: Dict = dict()
        self.owner = None

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


def custom_symbolic_tracer(root, concrete_args=None):
    tracer = CustomTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(
        root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


class CustomTracer(QuantizationTracer):

    def __init__(self,
                 skipped_methods=None,
                 skipped_module_names=None,
                 skipped_module_classes=None,
                 *args,
                 **kwargs):
        super(CustomTracer, self).__init__(skipped_module_names,
                                           skipped_module_classes)
        UntracedMethodRegistry.tracer = self
        self.skipped_methods = skipped_methods
        if self.skipped_methods:
            self.register_skipped_methods()

    @staticmethod
    def _check_valid_source(source):
        """Check if the source's format is valid."""
        if not isinstance(source, str):
            raise TypeError(f'source should be a str '
                            f'instance, but got {type(source)}')

        assert len(source.split('.')) > 1, \
            'source must have at least one `.`'

    def register_skipped_methods(self):
        if not isinstance(self.skipped_methods, list):
            self.skipped_methods = [self.skipped_methods]
        for s_method in self.skipped_methods:
            self._check_valid_source(s_method)
            mod_str = '.'.join(s_method.split('.')[:-2])
            cls_str = s_method.split('.')[-2]
            method_str = s_method.split('.')[-1]

            try:
                mod = import_modules_from_strings(mod_str)
            except ImportError:
                raise ImportError(f'{mod_str} is not imported correctly.')

            imported_cls: type = getattr(mod, cls_str)
            if not isinstance(imported_cls, type):
                raise TypeError(f'{cls_str} should be a type '
                                f'instance, but got {type(imported_cls)}')
            assert hasattr(imported_cls, method_str), \
                   f'{method_str} is not in {mod_str}.'

            method = getattr(imported_cls, method_str)

            method_registry = UntracedMethodRegistry(method)
            method_registry.__set_name__(imported_cls, method_str)

    def call_method(self, m: torch.nn.Module, name, method, args, kwargs):
        """Method that specifies the behavior of this ``Tracer`` when it
        encounters a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf
        module via ``is_leaf_module``. If it is, emit a ``call_module``
        node referring to ``m`` in the ``Graph``. Otherwise, call the
        ``Module`` normally, tracing through the operations in its ``forward``
        function.

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
        # if isinstance(root, (BaseClassifier, BaseDetector)):
        #     self.root = root
        #     fn = type(root).forward_trace
        #     self.submodule_paths = {
        #         mod: name
        #         for name, mod in root.named_modules()
        #     }
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

        # Reduce number of get_attr calls
        parameter_proxy_cache: Dict[str, Proxy] = {}

        # Method dispatch on parameters is not recorded unless it's directly
        # used. Thus, we need to insert a proxy when __getattr__ requests a
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

    def is_leaf_module(self, m: torch.nn.Module,
                       module_qualified_name: str) -> bool:
        # return super().is_leaf_module(m, module_qualified_name)
        leaf = super().is_leaf_module(m, module_qualified_name)
        return leaf
