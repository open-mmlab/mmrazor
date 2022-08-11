import torch
from torch.ao.quantization import FakeQuantize
from mmrazor.registry import MODELS
from mmrazor.models.utils import (_is_per_channel, _is_per_tensor, 
    _is_symmetric_quant, _is_float_qparams) 
from mmrazor.models.observers import MinMaxObserver


@MODELS.register_module()
class BaseFakeQuantize(FakeQuantize):
    def __init__(self, observer=MinMaxObserver()):
        super().__init__()
        self.activation_post_process = observer
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)
        
        bitrange = torch.tensor(self.quant_max - self.quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.is_pot_scale = self.activation_post_process.is_pot_scale
        self.is_symmetric_quant = _is_symmetric_quant(self.qscheme)


        