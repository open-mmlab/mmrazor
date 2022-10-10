# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch.nn

from mmrazor.registry import TASK_UTILS
from .base_estimator import BaseEstimator
from .counters import get_model_flops_params, get_model_latency


@TASK_UTILS.register_module()
class ResourceEstimator(BaseEstimator):
    """Estimator for calculating the resources consume.

    Args:
        input_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224).
        units (dict): Dict that contains converted FLOPs/params/latency units.
            Default to dict(flops='M', params='M', latency='ms').
        as_strings (bool): Output FLOPs/params/latency counts in a string
            form. Default to False.
        flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
            Default to None.
        latency_cfg (dict): Cfg for estimating latency. Default to None.

    Examples:
        >>> # direct calculate resource consume of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> estimator = ResourceEstimator(input_shape=(1, 3, 64, 64))
        >>> estimator.estimate(model=conv2d)
        {'flops': 3.444, 'params': 0.001, 'latency': 0.0}

        >>> # direct calculate resource consume of nn.Conv2d
        >>> conv2d = nn.Conv2d(3, 32, 3)
        >>> estimator = ResourceEstimator()
        >>> flops_params_cfg = dict(input_shape=(1, 3, 32, 32))
        >>> estimator.estimate(model=conv2d, flops_params_cfg)
        {'flops': 0.806, 'params': 0.001, 'latency': 0.0}

        >>> # calculate resources of custom modules
        >>> class CustomModule(nn.Module):
        ...
        ...    def __init__(self) -> None:
        ...        super().__init__()
        ...
        ...    def forward(self, x):
        ...        return x
        ...
        >>> @TASK_UTILS.register_module()
        ... class CustomModuleCounter(BaseCounter):
        ...
        ...    @staticmethod
        ...    def add_count_hook(module, input, output):
        ...        module.__flops__ += 1000000
        ...        module.__params__ += 700000
        ...
        >>> model = CustomModule()
        >>> flops_params_cfg = dict(input_shape=(1, 3, 64, 64))
        >>> estimator.estimate(model=model, flops_params_cfg)
        {'flops': 1.0, 'params': 0.7, 'latency': 0.0}
        ...
        >>> # calculate resources of custom modules with disable_counters
        >>> flops_params_cfg = dict(input_shape=(1, 3, 64, 64),
        ...                         disabled_counters=['CustomModuleCounter'])
        >>> estimator.estimate(model=model, flops_params_cfg)
        {'flops': 0.0, 'params': 0.0, 'latency': 0.0}

        >>> # calculate resources of mmrazor.models
        NOTE: check 'EstimateResourcesHook' in
              mmrazor.engine.hooks.estimate_resources_hook for details.
    """

    def __init__(
        self,
        input_shape: Tuple = (1, 3, 224, 224),
        units: Dict = dict(flops='M', params='M', latency='ms'),
        as_strings: bool = False,
        flops_params_cfg: Optional[dict] = None,
        latency_cfg: Optional[dict] = None,
    ):
        super().__init__(input_shape, units, as_strings)
        if not isinstance(units, dict):
            raise TypeError('units for estimator should be a dict',
                            f'but got `{type(units)}`')
        for unit_key in units:
            if unit_key not in ['flops', 'params', 'latency']:
                raise KeyError(f'Got invalid key `{unit_key}` in units. ',
                               'Should be `flops`, `params` or `latency`.')
        if flops_params_cfg:
            self.flops_params_cfg = flops_params_cfg
        else:
            self.flops_params_cfg = dict()
        self.latency_cfg = latency_cfg if latency_cfg else dict()

    def estimate(self,
                 model: torch.nn.Module,
                 flops_params_cfg: dict = None,
                 latency_cfg: dict = None) -> Dict[str, Union[float, str]]:
        """Estimate the resources(flops/params/latency) of the given model.

        This method will first parse the merged :attr:`self.flops_params_cfg`
        and the :attr:`self.latency_cfg` to check whether the keys are valid.

        Args:
            model: The measured model.
            flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
                Default to None.
            latency_cfg (dict): Cfg for estimating latency. Default to None.

            NOTE: If the `flops_params_cfg` and `latency_cfg` are both None,
            this method will only estimate FLOPs/params with default settings.

        Returns:
            Dict[str, Union[float, str]]): A dict that contains the resource
                results(FLOPs, params and latency).
        """
        resource_metrics = dict()
        measure_latency = True if latency_cfg else False

        if flops_params_cfg:
            flops_params_cfg = {**self.flops_params_cfg, **flops_params_cfg}
            self._check_flops_params_cfg(flops_params_cfg)
            flops_params_cfg = self._set_default_resource_params(
                flops_params_cfg)
        else:
            flops_params_cfg = self.flops_params_cfg

        if latency_cfg:
            latency_cfg = {**self.latency_cfg, **latency_cfg}
            self._check_latency_cfg(latency_cfg)
            latency_cfg = self._set_default_resource_params(latency_cfg)
        else:
            latency_cfg = self.latency_cfg

        model.eval()
        flops, params = get_model_flops_params(model, **flops_params_cfg)
        if measure_latency:
            latency = get_model_latency(model, **latency_cfg)
        else:
            latency = '0.0 ms' if self.as_strings else 0.0  # type: ignore

        resource_metrics.update({
            'flops': flops,
            'params': params,
            'latency': latency
        })
        return resource_metrics

    def estimate_separation_modules(
            self,
            model: torch.nn.Module,
            flops_params_cfg: dict = None) -> Dict[str, Union[float, str]]:
        """Estimate FLOPs and params of the spec modules with separate return.

        Args:
            model: The measured model.
            flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
                Default to None.

        Returns:
            Dict[str, Union[float, str]]): A dict that contains the FLOPs and
                params results (string | float format) of each modules in the
                ``flops_params_cfg['spec_modules']``.
        """
        if flops_params_cfg:
            flops_params_cfg = {**self.flops_params_cfg, **flops_params_cfg}
            self._check_flops_params_cfg(flops_params_cfg)
            flops_params_cfg = self._set_default_resource_params(
                flops_params_cfg)
        else:
            flops_params_cfg = self.flops_params_cfg
        flops_params_cfg['seperate_return'] = True

        assert len(flops_params_cfg['spec_modules']), (
            'spec_modules can not be empty when calling '
            f'`estimate_separation_modules` of {self.__class__.__name__} ')

        model.eval()
        spec_modules_resources = get_model_flops_params(
            model, **flops_params_cfg)
        return spec_modules_resources

    def _check_flops_params_cfg(self, flops_params_cfg: dict) -> None:
        """Check the legality of ``flops_params_cfg``.

        Args:
            flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
        """
        for key in flops_params_cfg:
            if key not in get_model_flops_params.__code__.co_varnames[
                    1:]:  # type: ignore
                raise KeyError(f'Got invalid key `{key}` in flops_params_cfg.')

    def _check_latency_cfg(self, latency_cfg: dict) -> None:
        """Check the legality of ``latency_cfg``.

        Args:
            latency_cfg (dict): Cfg for estimating latency.
        """
        for key in latency_cfg:
            if key not in get_model_latency.__code__.co_varnames[
                    1:]:  # type: ignore
                raise KeyError(f'Got invalid key `{key}` in latency_cfg.')

    def _set_default_resource_params(self, cfg: dict) -> dict:
        """Set default attributes for the input cfgs.

        Args:
            cfg (dict): flops_params_cfg or latency_cfg.
        """
        default_common_settings = ['input_shape', 'units', 'as_strings']
        for key in default_common_settings:
            if key not in cfg:
                cfg[key] = getattr(self, key)
        return cfg
