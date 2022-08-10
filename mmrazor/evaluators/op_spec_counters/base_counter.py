# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod


class BaseCounter(object, metaclass=ABCMeta):
    """Base class of all `OP_SPEC_COUNTERS`"""

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractclassmethod
    def add_count_hook(module, input, output):
        """the main func of a `BaseCounter`

        Args:
            module (nn.Module): the module to be tested.
            input (_type_): input_tensor. Plz refer to `torch forward_hook`
            output (_type_): output_tensor. Plz refer to `torch forward_hook`
        """
        pass
