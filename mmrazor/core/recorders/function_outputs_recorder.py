# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import FunctionType
from typing import List, Optional

from mmcv.utils import import_modules_from_strings
from torch import nn

from ..builder import RECORDERS
from .base_recorder import BaseRecorder


@RECORDERS.register_module()
class FunctionOutputsRecorder(BaseRecorder):
    """Recorder for intermediate results which are ``FunctionType``'s outputs.

    Args:
        sources (List(str)): The names of the functions whose output
            needs  to be recorded.
        import_modules (List(str)): The modules which source
            functions belong to.

    TODO the example is not suitable enough.
    Examples:
        >>> import copy
        >>> import torch
        >>> from mmcv.cnn import ConvModule
        >>> from mmrazor.core import build_recorder

        >>> # example recorder config
        >>> recorder_cfg = dict(
        >>>     type='FunctionOutputs',
        >>>     sources=['build_conv_layer'],
        >>>     import_modules=['mmcv.cnn.bricks.conv_module'])

        >>> recorder_cfg_ = copy.deepcopy(recorder_cfg)
        >>> recorder_cfg_['type'] = recorder_cfg['type'] + 'Recorder'
        >>> ctx = build_recorder(recorder_cfg_)
        >>> ctx.initialize()

        >>> with ctx:
        >>>     conv = ConvModule(1,1,1)

        >>> ctx.data_buffer
        >>> {'mmcv.cnn.bricks.conv_module.build_conv_layer':
        >>>     [Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))]
    """

    def __init__(self, sources: List, import_modules: List):
        super().__init__()
        self.sources: List(str) = sources
        self.import_modules: List(str) = import_modules

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Wrapper the origin source functions.

        The ``model`` is useless in this recorder, just to be consistent with
        other recorders.
        """
        for func_name, module_name in zip(self.sources, self.import_modules):
            if module_name is None:
                origin_func = eval(f'{func_name}')
            else:
                # import the function corrosponding module
                imported_module = import_modules_from_strings(  # noqa: F841
                    module_name)
                origin_func = eval(f'imported_module.{func_name}')
            buffer_key = f'{module_name}.{func_name}'
            # init data_buffer, data_buffer must be Dict(str, list)
            # assume a func execute N times, there will be N outputs need to
            # save.
            self.data_buffer[buffer_key] = list()
            # add record wrapper to origin function.
            wrapped_func = self.func_record_wrapper(  # noqa: F841
                origin_func, buffer_key)

            if module_name is None:
                exec(f'{func_name} = wrapped_func')
            else:
                exec(f'imported_module.{func_name} = wrapped_func')

    def func_record_wrapper(self, origin_func: FunctionType,
                            buffer_key: str) -> FunctionType:
        """Save the function's outputs.

        Args:
            origin_func (FunctionType): The method whose outputs need to be
                recorded.
            buffer_key (str): The key of the function's outputs saved in
                ``data_buffer``.
        """

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):
            outputs = origin_func(*args, **kwargs)
            if self.recording:
                self.data_buffer[buffer_key].append(outputs)
            return outputs

        return wrap_func
