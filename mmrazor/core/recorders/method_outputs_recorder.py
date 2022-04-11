# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import MethodType
from typing import Optional, Any, Dict, List
from torch import nn
from mmcv.utils import import_modules_from_strings

from ..builder import RECORDERS
from .base_recorder import BaseRecorder


@RECORDERS.register_module()
class MethodOutputsRecorder(BaseRecorder):
    """Recorder for intermediate results which are ``MethodType``'s outputs.
    
    Note:
        Different from ``FunctionType``, ``MethodType`` is the type of methods 
        of class instances.

    Args:
        sources (List(str)): The names of the methods whose output needs to be 
            recorded.
        import_modules (List(str)): The modules which source methods belong to. 
    
    Examples:
        >>> import copy
        >>> import torch
        >>> from mmcv.cnn import ConvModule
        >>> from mmrazor.core import build_recorder

        >>> # example recorder config
        >>> recorder_cfg = dict(
        >>>     type='MethodOutputs',
        >>>     sources=['ConvModule.forward'],
        >>>     import_modules=['mmcv.cnn'])
    
        >>> recorder_cfg_ = copy.deepcopy(recorder_cfg)
        >>> recorder_cfg_['type'] = recorder_cfg['type'] + 'Recorder'
        >>> ctx = build_recorder(recorder_cfg_)
        >>> ctx.initialize()

        >>> conv = ConvModule(1,1,1)
        >>> with ctx:
        >>>     res = conv(torch.randn(1,1,1,1))

        >>> ctx.data_buffer
        >>> {'mmcv.cnn.ConvModule.forward': [tensor([[[[0.]]]])]}
        >>> ctx.get_record_data('mmcv.cnn.ConvModule.forward')
        >>> [tensor([[[[0.]]]])]
        >>> ctx.get_record_data('mmcv.cnn.ConvModule.forward', data_index=0)
        >>> tensor([[[[0.]]]])
    """
    def __init__(self, sources: List, import_modules: List):
        super().__init__()
        self.sources: List(str) = sources
        self.import_modules: List(str) = import_modules

    def prepare_from_model(self, model: nn.Module) -> None:
        """Wrapper the origin source methods.
        The ``model`` is useless in this recorder, just to be consistent with 
        other recorders.
        """

        for method_name, module_name in zip(self.sources, self.import_modules):
            
            if module_name is None:
                origin_method = eval(f'{method_name}')
            else:
                # import the method corrosponding module
                imported_module = import_modules_from_strings(  # noqa: F841
                    module_name)
                origin_method = eval(f'imported_module.{method_name}')

            buffer_key = f'{module_name}.{method_name}'
            # init data_buffer, data_buffer must be Dict(str, list)
            # assume a method execute N times, there will be N outputs need to 
            # save.
            self.data_buffer[buffer_key] = list()
            # add record wrapper to origin method.
            wrapped_method = self.method_record_wrapper(  # noqa: F841
                origin_method, buffer_key)
            
            if module_name is None:
                exec(f'{method_name} = wrapped_method')
            else:
                exec(f'imported_module.{method_name} = wrapped_method')

    def method_record_wrapper(self, 
                              orgin_method: MethodType, 
                              buffer_key: str) -> MethodType:
        """Save the method's outputs.
        
        Args:
            origin_method (MethodType): The method whose outputs need to be 
                recorded.
            buffer_key (str): The key of the method's outputs saved in 
                ``data_buffer``.
        """

        @functools.wraps(orgin_method)
        def wrap_method(*args, **kwargs):
            outputs = orgin_method(*args, **kwargs)
            if self.recording:
                self.data_buffer[buffer_key].append(outputs)
            return outputs

        return wrap_method
