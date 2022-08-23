# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod


class BaseCounter(object, metaclass=ABCMeta):
    """Base class of all op module counters in `TASK_UTILS`.

    In ResourceEstimator, `XXModuleCounter` is responsible for `XXModule`,
    which refers to estimator/flops_params_counter.py::get_counter_type().
    Users can customize a `ModuleACounter` and overwrite the `add_count_hook`
    method with a self-defined module `ModuleA`.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractclassmethod
    def add_count_hook(module, input, output):
        """The main method of a `BaseCounter` which defines the way to
        calculate resources(flops/params) of the current module.

        Args:
            module (nn.Module): the module to be tested.
            input (_type_): input_tensor. Plz refer to `torch forward_hook`
            output (_type_): output_tensor. Plz refer to `torch forward_hook`
        """
        pass
