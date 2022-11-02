# Copyright (c) OpenMMLab. All rights reserved.
import torch


class QuantizeScheme(object):
    """Describe quantization scheme."""

    def __init__(self,
                 bit=8,
                 is_symmetry=True,
                 is_per_channel=False,
                 is_pot_scale=False,
                 **kwargs):
        self.bit = bit
        self.is_symmetry = is_symmetry
        self.is_per_channel = is_per_channel
        self.is_pot_scale = is_pot_scale

        if self.is_per_channel:
            self.torch_qscheme = torch.per_channel_symmetric \
                if self.is_symmetry else torch.per_channel_affine
        else:
            self.torch_qscheme = torch.per_tensor_symmetric \
                if self.is_symmetry else torch.per_tensor_affine
        if 'is_symmetric_range' in kwargs:
            self.is_symmetric_range = kwargs['is_symmetric_range']
            del kwargs['is_symmetric_range']
        else:
            self.is_symmetric_range = False
        self.kwargs = kwargs

    def to_observer_params(self):
        quant_min = 0
        quant_max = 2**self.bit - 1
        if self.is_symmetry:
            quant_max = 2**(self.bit - 1) - 1
            if self.is_symmetric_range:
                quant_min = -2**(self.bit - 1) + 1
            else:
                quant_min = -2**(self.bit - 1)

        naive_para = {
            'quant_min': quant_min,
            'quant_max': quant_max,
            'dtype': torch.qint8 if self.is_symmetry else torch.quint8,
            'is_pot_scale': self.is_pot_scale,
            'qscheme': self.torch_qscheme,
            'reduce_range': False,
            'ch_axis': 0 if self.is_per_channel else -1
        }
        naive_para.update(self.kwargs)
        return naive_para

    def __str__(self):
        return f'bit: {self.bit} / is_symmetry: {self.is_symmetry} / \
                is_per_channel: {self.is_per_channel} / is_pot_scale: \
                {self.is_pot_scale} / extra_kwargs: {self.kwargs}'
