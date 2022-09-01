# Copyright (c) OpenMMLab. All rights reserved.
import sys
from functools import partial

import torch
import torch.nn as nn

from mmrazor.registry import TASK_UTILS


def get_model_complexity_info(model,
                              input_shape=(1, 3, 224, 224),
                              spec_modules=[],
                              disabled_counters=[],
                              print_per_layer_stat=False,
                              as_strings=False,
                              input_constructor=None,
                              flush=False,
                              ost=sys.stdout):
    """Get complexity information of a model. This method can calculate FLOPs
    and parameter counts of a model with corresponding input shape. It can also
    print complexity information for each layer in a model. Supported layers
    are listed as below:

        - Convolutions: ``nn.Conv1d``, ``nn.Conv2d``, ``nn.Conv3d``.
        - Activations: ``nn.ReLU``, ``nn.PReLU``, ``nn.ELU``, ``nn.LeakyReLU``,
            ``nn.ReLU6``.
        - Poolings: ``nn.MaxPool1d``, ``nn.MaxPool2d``, ``nn.MaxPool3d``,
            ``nn.AvgPool1d``, ``nn.AvgPool2d``, ``nn.AvgPool3d``,
            ``nn.AdaptiveMaxPool1d``, ``nn.AdaptiveMaxPool2d``,
            ``nn.AdaptiveMaxPool3d``, ``nn.AdaptiveAvgPool1d``,
            ``nn.AdaptiveAvgPool2d``, ``nn.AdaptiveAvgPool3d``.
        - BatchNorms: ``nn.BatchNorm1d``, ``nn.BatchNorm2d``,
            ``nn.BatchNorm3d``.
        - Linear: ``nn.Linear``.
        - Deconvolution: ``nn.ConvTranspose2d``.
        - Upsample: ``nn.Upsample``.

    Args:
        model (nn.Module): The model for complexity calculation.
        input_shape (tuple): Input shape (including batchsize) used for
            calculation. Default to (1, 3, 224, 224)
        spec_modules (list): A list that contains the names of several spec
            modules, which users want to get resources infos of them.
            e.g., ['backbone', 'head'], ['backbone.layer1']. Default to [].
        disabled_counters (list): One can limit which ops' spec would be
            calculated. Default to [].
        print_per_layer_stat (bool): Whether to print complexity information
            for each layer in a model. Default to True.
        as_strings (bool): Output FLOPs and params counts in a string form.
            Default to True.
        input_constructor (None | callable): If specified, it takes a callable
            method that generates input. otherwise, it will generate a random
            tensor with input shape to calculate FLOPs. Default to None.
        flush (bool): same as that in :func:`print`. Default to False.
        ost (stream): same as ``file`` param in :func:`print`.
            Default to sys.stdout.

    Returns:
        tuple[float | str] | dict[str, float]: If `as_strings` is set to True,
            it will return FLOPs and parameter counts in a string format.
            Otherwise, it will return those in a float number format.
            If len(spec_modules) > 0, it will return a resource info dict with
            FLOPs and parameter counts of each spec module in float format.
    """
    assert type(input_shape) is tuple
    assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)
    flops_params_model = add_flops_params_counting_methods(model)
    flops_params_model.eval()
    flops_params_model.start_flops_params_count(disabled_counters)
    if input_constructor:
        input = input_constructor(input_shape)
        _ = flops_params_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                tuple(input_shape),
                dtype=next(flops_params_model.parameters()).dtype,
                device=next(flops_params_model.parameters()).device)
        except StopIteration:
            # Avoid StopIteration for models which have no parameters,
            # like `nn.Relu()`, `nn.AvgPool2d`, etc.
            batch = torch.ones(()).new_empty(tuple(input_shape))

        _ = flops_params_model(batch)

    flops_count, params_count = \
        flops_params_model.compute_average_flops_params_cost()

    if print_per_layer_stat:
        print_model_with_flops_params(
            flops_params_model,
            flops_count,
            params_count,
            ost=ost,
            flush=flush)

    if len(spec_modules):
        module_names = [name for name, _ in flops_params_model.named_modules()]
        for module in spec_modules:
            assert module in module_names, \
                f'All modules in spec_modules should be in the measured ' \
                f'flops_params_model. Got module {module} in spec_modules.'
        spec_modules_resources = dict()
        accumulate_sub_module_flops_params(flops_params_model)
        for name, module in flops_params_model.named_modules():
            if name in spec_modules:
                spec_modules_resources[name] = dict()
                spec_modules_resources[name]['flops'] = module.__flops__
                spec_modules_resources[name]['params'] = module.__params__
                if as_strings:
                    spec_modules_resources[name]['flops'] = str(
                        params_units_convert(module.__flops__,
                                             'G')) + ' GFLOPs'
                    spec_modules_resources[name]['params'] = str(
                        params_units_convert(module.__params__, 'M')) + ' M'

    flops_params_model.stop_flops_params_count()

    if len(spec_modules):
        return spec_modules_resources

    if as_strings:
        flops_string = str(params_units_convert(flops_count, 'G')) + ' GFLOPs'
        params_string = str(params_units_convert(params_count, 'M')) + ' M'
        return flops_string, params_string

    return flops_count, params_count


def params_units_convert(num_params, units='M', precision=3):
    """Convert parameter number with units.

    Args:
        num_params (float): Parameter number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'M',
            'K' and ''. If set to None, it will automatically choose the most
            suitable unit for Parameter number. Default to None.
        precision (int): Digit number after the decimal point. Default to 2.

    Returns:
        str: The converted parameter number.

    Examples:
        >>> params_units_convert(1e9)
        '1000.0'
        >>> params_units_convert(2e5)
        '200.0'
        >>> params_units_convert(3e-9)
        '3e-09'
    """

    if units == 'G':
        return round(num_params / 10.**9, precision)
    elif units == 'M':
        return round(num_params / 10.**6, precision)
    elif units == 'K':
        return round(num_params / 10.**3, precision)
    else:
        raise ValueError(f'Unsupported units convert: {units}')


def print_model_with_flops_params(model,
                                  total_flops,
                                  total_params,
                                  units='G',
                                  precision=3,
                                  ost=sys.stdout,
                                  flush=False):
    """Print a model with FLOPs and Params for each layer.

    Args:
        model (nn.Module): The model to be printed.
        total_flops (float): Total FLOPs of the model.
        total_params (float): Total parameter counts of the model.
        units (str | None): Converted FLOPs units. Default to 'G'.
        precision (int): Digit number after the decimal point. Default to 3.
        ost (stream): same as `file` param in :func:`print`.
            Default to sys.stdout.
        flush (bool): same as that in :func:`print`. Default to False.

    Example:
        >>> class ExampleModel(nn.Module):
        >>> def __init__(self):
        >>>     super().__init__()
        >>>     self.conv1 = nn.Conv2d(3, 8, 3)
        >>>     self.conv2 = nn.Conv2d(8, 256, 3)
        >>>     self.conv3 = nn.Conv2d(256, 8, 3)
        >>>     self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        >>>     self.flatten = nn.Flatten()
        >>>     self.fc = nn.Linear(8, 1)
        >>> def forward(self, x):
        >>>     x = self.conv1(x)
        >>>     x = self.conv2(x)
        >>>     x = self.conv3(x)
        >>>     x = self.avg_pool(x)
        >>>     x = self.flatten(x)
        >>>     x = self.fc(x)
        >>>     return x
        >>> model = ExampleModel()
        >>> x = (3, 16, 16)
        to print the complexity information state for each layer, you can use
        >>> get_model_complexity_info(model, x)
        or directly use
        >>> print_model_with_flops_params(model, 4579784.0, 37361)
        ExampleModel(
          0.037 M, 100.000% Params, 0.005 GFLOPs, 100.000% FLOPs,
          (conv1): Conv2d(0.0 M, 0.600% Params, 0.0 GFLOPs, 0.959% FLOPs, 3, 8, kernel_size=(3, 3), stride=(1, 1))  # noqa: E501
          (conv2): Conv2d(0.019 M, 50.020% Params, 0.003 GFLOPs, 58.760% FLOPs, 8, 256, kernel_size=(3, 3), stride=(1, 1))
          (conv3): Conv2d(0.018 M, 49.356% Params, 0.002 GFLOPs, 40.264% FLOPs, 256, 8, kernel_size=(3, 3), stride=(1, 1))
          (avg_pool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.017% FLOPs, output_size=(1, 1))
          (flatten): Flatten(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          (fc): Linear(0.0 M, 0.024% Params, 0.0 GFLOPs, 0.000% FLOPs, in_features=8, out_features=1, bias=True)
        )
    """

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_num_params = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        flops_string = str(
            params_units_convert(
                accumulated_flops_cost, units=units,
                precision=precision)) + ' ' + units + 'FLOPs'
        params_string = str(
            params_units_convert(
                accumulated_num_params, units='M', precision=precision)) + ' M'
        return ', '.join([
            params_string,
            '{:.3%} Params'.format(accumulated_num_params / total_params),
            flops_string,
            '{:.3%} FLOPs'.format(accumulated_flops_cost / total_flops),
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost, flush=flush)
    model.apply(del_extra_repr)


def accumulate_sub_module_flops_params(model):
    """Accumulate FLOPs and params for each module in the model. Each module in
    the model will have the `__flops__` and `__params__` parameters.

    Args:
        model (nn.Module): The model to be accumulated.
    """

    def accumulate_params(module):
        if is_supported_instance(module):
            return module.__params__
        else:
            sum = 0
            for m in module.children():
                sum += accumulate_params(m)
            return sum

    def accumulate_flops(module):
        if is_supported_instance(module):
            return module.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in module.children():
                sum += accumulate_flops(m)
            return sum

    for module in model.modules():
        _flops = accumulate_flops(module)
        _params = accumulate_params(module)
        module.__flops__ = _flops
        module.__params__ = _params


def get_model_parameters_number(model):
    """Calculate parameter number of a model.

    Args:
        model (nn.module): The model for parameter number calculation.
    Returns:
        float: Parameter number of the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def add_flops_params_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_params_count = start_flops_params_count.__get__(  # noqa: E501
        net_main_module)
    net_main_module.stop_flops_params_count = stop_flops_params_count.__get__(
        net_main_module)
    net_main_module.reset_flops_params_count = reset_flops_params_count.__get__(  # noqa: E501
        net_main_module)
    net_main_module.compute_average_flops_params_cost = compute_average_flops_params_cost.__get__(  # noqa: E501
        net_main_module)

    net_main_module.reset_flops_params_count()

    return net_main_module


def compute_average_flops_params_cost(self):
    """Compute average FLOPs and Params cost.

    A method to compute average FLOPs cost, which will be available after
    `add_flops_params_counting_methods()` is called on a desired net object.
    Returns:
        float: Current mean flops consumption per image.
    """
    batches_count = self.__batch_counter__
    flops_sum = 0
    params_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
            params_sum += module.__params__
    return flops_sum / batches_count, params_sum


def start_flops_params_count(self, disabled_counters):
    """Activate the computation of mean flops and params consumption per image.

    A method to activate the computation of mean flops consumption per image.
    which will be available after ``add_flops_params_counting_methods()`` is
    called on a desired net object. It should be called before running the
    network.
    """
    add_batch_counter_hook_function(self)

    def add_flops_params_counter_hook_function(module):
        if is_supported_instance(module):
            if hasattr(module, '__flops_params_handle__'):
                return

            else:
                counter_type = get_counter_type(module)
                if (disabled_counters is None
                        or counter_type not in disabled_counters):
                    counter = TASK_UTILS.build(
                        dict(type=counter_type, _scope_='mmrazor'))
                    handle = module.register_forward_hook(
                        counter.add_count_hook)

                    module.__flops_params_handle__ = handle
                else:
                    return

    self.apply(partial(add_flops_params_counter_hook_function))


def stop_flops_params_count(self):
    """Stop computing the mean flops and params consumption per image.

    A method to stop computing the mean flops consumption per image, which will
    be available after ``add_flops_params_counting_methods()`` is called on a
    desired net object. It can be called to pause the computation whenever.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_params_counter_hook_function)


def reset_flops_params_count(self):
    """Reset statistics computed so far.

    A method to Reset computed statistics, which will be available after
    `add_flops_params_counting_methods()` is called on a desired net object.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_params_counter_variable_or_reset)


# ---- Internal functions
def empty_flops_params_counter_hook(module, input, output):
    module.__flops__ += 0
    module.__params__ += 0


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module, '
              'assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_params_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
        module.__flops__ = 0
        module.__params__ = 0


def get_counter_type(module):
    return module.__class__.__name__ + 'Counter'


def is_supported_instance(module):
    if get_counter_type(module) in TASK_UTILS._module_dict.keys():
        return True
    return False


def remove_flops_params_counter_hook_function(module):
    if hasattr(module, '__flops_params_handle__'):
        module.__flops_params_handle__.remove()
        del module.__flops_params_handle__
    if hasattr(module, '__flops__'):
        del module.__flops__
    if hasattr(module, '__params__'):
        del module.__params__
