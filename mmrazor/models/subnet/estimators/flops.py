# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
from functools import wraps
from typing import IO, Callable, Dict, Iterable, Optional, Tuple, Type

from mmcv.cnn.utils import flops_counter as mmcv_flops_counter
from mmcv.cnn.utils import get_model_complexity_info
from torch.nn import Module

from ..fix_subnet import VALID_FIX_MUTABLE_TYPE, load_fix_subnet


class FlopsEstimator:
    """An estimator to help calculate flops of module.

    FlopsEstimator is based on flops counter in mmcv, it can be directly used
    to calculate flops of module or calculate flops of subnet. Also, it
    provides api for adding flops counter hook to custom modules that are not
    supported by mmcv.

    Examples:
        >>> # direct calculate flops of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> FlopsEstimator.get_model_complexity_info(
        ...     conv2d,
        ...     input_shape=[3, 224, 224],
        ...     print_per_layer_stat=False)
        ('0.04 GFLOPs', '896')

        >>> # calculate flops of custom modules
        >>> class FoolAddConstant(nn.Module):
        ...
        ...     def __init__(self, p: float = 0.1) -> None:
        ...         super().__init__()
        ...
        ...         self.register_parameter(
        ...             name='p',
        ...             param=Parameter(torch.tensor(p, dtype=torch.float32)))
        ...
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return x + self.p
        ...
        >>> def fool_add_constant_flops_counter_hook(
        ...         add_constant_module: nn.Module,
        ...         input: Tensor,
        ...         output: Tensor) -> None:
        ...     add_constant_module.__flops__ = 1e8
        ...
        >>> FlopsEstimator.register_module(
        ...     flops_counter_hook=fool_add_constant_flops_counter_hook,
        ...     module=FoolAddConstant)
        >>> model = FoolAddConstant()
        >>> FlopsEstimator.get_model_complexity_info(
        ...     model=model,
        ...     input_shape=[3, 224, 224],
        ...     print_per_layer_stat=False)
        ('0.1 GFLOPs', '1')

        >>> # calculate subnet flops
        >>> class FoolOneShotModel(nn.Module):
        ...
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...
        ...         candidates = nn.ModuleDict({
        ...             'conv3x3': nn.Conv2d(3, 32, 3),
        ...             'conv5x5': nn.Conv2d(3, 32, 5)})
        ...         self.op = OneShotMutableOP(candidates)
        ...         self.op.current_choice = 'conv3x3'
        ...
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return self.op(x)
        ...
        >>> model = FoolOneShotModel()
        >>> fix_subnet = export_fix_subnet(model)
        >>> fix_subnet
        FixSubnet(modules={'op': 'conv3x3'}, channels=None)
        >>> FlopsEstimator.get_model_complexity_info(
        ...     supernet=model,
        ...     fix_subnet=fix_subnet,
        ...     input_shape=[3, 224, 224],
        ...     print_per_layer_stat=False)
        ('0.04 GFLOPs', '896')
    """

    _mmcv_modules_mapping: Dict[Module, Callable] = \
        mmcv_flops_counter.get_modules_mapping()
    _custom_modules_mapping: Dict[Module, Callable] = {}

    @staticmethod
    def get_model_complexity_info(
            model: Module,
            fix_mutable: Optional[VALID_FIX_MUTABLE_TYPE] = None,
            input_shape: Iterable[int] = (3, 224, 224),
            input_constructor: Optional[Callable] = None,
            print_per_layer_stat: bool = True,
            as_strings: bool = True,
            flush: bool = False,
            ost: IO = sys.stdout) -> Tuple:
        """Get complexity information of model.

        This method is based on ``get_model_complexity_info`` of mmcv. It can
        calculate FLOPs and parameter counts of a model with corresponding
        input shape. It can also print complexity information for each layer
        in a model.

        Args:
            model (torch.nn.Module): The model for complexity calculation.
            fix_mutable (VALID_FIX_MUTABLE_TYPE, optional): The config of fixed
                subnet. When this argument is specified, the function will
                return complexity information of the subnet. Default: None.
            input_shape (Iterable[int]): Input shape used for calculation.
            print_per_layer_stat (bool): Whether to print complexity
                information for each layer in a model. Default: True.
            as_strings (bool): Output FLOPs and params counts in a string form.
                Default: True.
            input_constructor (Callable, optional): If specified, it takes a
                callable method that generates input. otherwise, it will
                generate a random tensor with input shape to calculate FLOPs.
                Default: None.
            flush (bool): same as that in :func:`print`. Default: False.
            ost (stream): same as ``file`` param in :func:`print`.
                Default: sys.stdout.

        Returns:
            tuple[float | str]: If ``as_strings`` is set to True, it will
                return FLOPs and parameter counts in a string format.
                otherwise, it will return those in a float number format.
        """
        copied_model = copy.deepcopy(model)
        if fix_mutable is not None:
            load_fix_subnet(copied_model, fix_mutable)

        return get_model_complexity_info(
            model=copied_model,
            input_shape=input_shape,
            input_constructor=input_constructor,
            print_per_layer_stat=print_per_layer_stat,
            as_strings=as_strings,
            flush=flush,
            ost=ost)

    @classmethod
    def register_module(cls,
                        flops_counter_hook: Callable,
                        module: Optional[Type[Module]] = None,
                        force: bool = False) -> Optional[Callable]:
        """Register a module with flops_counter_hook.

        Args:
            flops_counter (Callable): The hook that specifies how to calculate
                flops of given module.
            module (torch.nn.Module, optional): Module class to be registered.
                Defaults to None.
            force (bool): Whether to override an existing flops_counter_hook
                with the same module. Default to False.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        if not (module is None or issubclass(module, Module)):
            raise TypeError(
                'module must be None, an subclass of torch.nn.Module, '
                f'but got {module}')

        if not callable(flops_counter_hook):
            raise TypeError('flops_counter_hook must be Callable, '
                            f'but got {type(flops_counter_hook)}')

        if module is not None:
            return cls._register_module(
                module=module,
                flops_counter_hook=flops_counter_hook,
                force=force)

        def _register(module: Type[Module]) -> None:
            cls._register_module(
                module=module,
                flops_counter_hook=flops_counter_hook,
                force=force)

        return _register

    @classmethod
    def remove_custom_module(cls, module: Type[Module]) -> None:
        """Remove a registered module.

        Args:
            module (torch.nn.Module): Module class to be removed.
        """
        if module not in cls._custom_modules_mapping:
            raise KeyError(f'{module} not in custom module mapping')

        del cls._custom_modules_mapping[module]

    @classmethod
    def clear_custom_module(cls) -> None:
        """Remove all registered modules."""
        cls._custom_modules_mapping.clear()

    @classmethod
    def _register_module(cls,
                         flops_counter_hook: Callable,
                         module: Type[Module],
                         force: bool = False) -> None:
        """Register a module with flops_counter_hook.

        Args:
            flops_counter (Callable): The hook that specifies how to calculate
                flops of given module.
            module (torch.nn.Module, optional): Module class to be registered.
                Defaults to None.
            force (bool): Whether to override an existing flops_counter_hook
                with the same module. Default to False.
        """
        if not force and module in cls.get_modules_mapping():
            raise KeyError(f'{module} is already registered')
        cls._custom_modules_mapping[module] = flops_counter_hook

    @classmethod
    def get_modules_mapping(cls) -> Dict[Module, Callable]:
        """Get all modules with their corresponding flops counter hook.

        Returns:
            Dict[Module, Callable]: Modules with their corresponding flops
                counter hook.
        """
        return {**cls._mmcv_modules_mapping, **cls._custom_modules_mapping}

    @classmethod
    def get_custom_modules_mapping(cls) -> Dict[Module, Callable]:
        """Get customed modules with their corresponding flops counter hook.

        Returns:
            Dict[Module, Callable]: Modules with their corresponding flops
                counter hook.
        """
        return {**cls._custom_modules_mapping}

    @classmethod
    def _mmcv_modules_mappings_wrapper(
            cls, mmcv_get_modules_mapping: Callable) -> Callable:
        """Wrapper for ``get_modules_mapping`` function in mmcv.

        Args:
            mmcv_get_modules_mapping (Callable): ``get_modules_mapping``
                function in mmcv.

        Returns:
            Callable: Wrapped ``get_modules_mapping`` function.
        """

        @wraps(mmcv_get_modules_mapping)
        def wrapper() -> Dict[Module, Callable]:
            mmcv_modules_mapping: Dict[Module, Callable] = \
                mmcv_get_modules_mapping()

            # TODO
            # use | operator
            # | operator only be supported in python 3.9.0 or greater
            return {**mmcv_modules_mapping, **cls._custom_modules_mapping}

        return wrapper


mmcv_flops_counter.get_modules_mapping = \
    FlopsEstimator._mmcv_modules_mappings_wrapper(
        mmcv_flops_counter.get_modules_mapping)
