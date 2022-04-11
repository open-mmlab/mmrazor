# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Any, Dict, List
from torch import nn
from ..builder import RECORDERS
from .base_recorder import BaseRecorder


@RECORDERS.register_module()
class ParameterRecorder(BaseRecorder):
    """Recorder for Pytorch model's parameters.
    
    Args:
        sources List(str): The names of the parameters need to be recorded.
    
    Examples:
        >>> import copy
        >>> import torch
        >>> from mmcv.cnn import ConvModule
        >>> from mmrazor.core import build_recorder

        >>> # example recorder config
        >>> recorder_cfg = dict(
        >>>     type='Parameter',
        >>>     sources=['conv.weight'])
    
        >>> recorder_cfg_ = copy.deepcopy(recorder_cfg)
        >>> recorder_cfg_['type'] = recorder_cfg['type'] + 'Recorder'
        >>> ctx = build_recorder(recorder_cfg_)

        >>> conv = ConvModule(1,1,1)
        >>> ctx.initialize(conv)

        >>> ctx.data_buffer
        >>> {'conv.weight': [Parameter containing: 
              tensor([[[[0.3244]]]], requires_grad=True)]}
        >>> ctx.get_record_data('conv.weight')
        >>> [Parameter containing: tensor([[[[0.3244]]]], requires_grad=True)]
        >>> ctx.get_record_data('conv.weight', data_index=0)
        >>> Parameter containing: tensor([[[[0.3244]]]], requires_grad=True)
    """
    def __init__(self, sources: List):
        super().__init__()
        self.sources: List(str) = sources

    def prepare_from_model(self, model: nn.Module) -> None:
        """Record the Pytorch model's parameters."""
        for param_name, param in model.named_parameters():
            if param_name in self.sources:
                # init data_buffer, data_buffer must be Dict(str, list)
                # param is not necessary saved in List,  just to be consistent 
                # with other recorders.
                self.data_buffer[param_name] = [param]

    def reset_data_buffer(self):
        """Clear data in data_buffer.
        
        Note:
            The data_buffer stores the address of the parameter in memory and 
            does not need to be reset
        """
        pass
