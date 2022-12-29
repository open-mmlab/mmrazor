# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import torch
from mmengine.config import Config
from torch.ao.quantization import QConfig

from mmrazor.registry import MODELS

RequiredArgs = [
    'w_qscheme', 'a_qscheme', 'w_fake_quant', 'a_fake_quant', 'w_observer',
    'a_observer'
]


class QConfigHander():
    """Convert custom user-friendly qconfig format to torch's QConfig.

    Args:
        qconfig (Dict | Config): custom user-friendly qconfig format,
            including setting observers, fakequants and quantization schemes
            for weights and activations.
    Note:
        whether quantization scheme is per-channel or not depends on
        used observer, if observer support per-channel quantization, its name
        should contain 'PerChannel'.
    """

    def __init__(self, qconfig: Union[Dict, Config]):
        if not self.check_qconfig(qconfig):
            raise ValueError('The format of qconfig is incorrect.')
        else:
            w_observer = MODELS.get(qconfig['w_observer']['type'])
            a_observer = MODELS.get(qconfig['a_observer']['type'])
            w_is_per_channel = False
            a_is_per_channel = False
            # import pdb;pdb.set_trace()
            if 'PerChannel' in w_observer.__name__:
                w_is_per_channel = True
            if 'PerChannel' in a_observer.__name__:
                a_is_per_channel = True
            self.w_qscheme = QSchemeHander(
                is_per_channel=w_is_per_channel, **qconfig['w_qscheme'])
            self.a_qscheme = QSchemeHander(
                is_per_channel=a_is_per_channel, **qconfig['a_qscheme'])

            w_fake_quant = MODELS.get(qconfig['w_fake_quant']['type'])
            w_observer_kwargs = self.w_qscheme.to_observer_params()
            a_fake_quant = MODELS.get(qconfig['a_fake_quant']['type'])
            a_observer_kwargs = self.a_qscheme.to_observer_params()

            self.w_fake_quant = w_fake_quant.with_args(
                observer=w_observer, **w_observer_kwargs)
            self.a_fake_quant = a_fake_quant.with_args(
                observer=a_observer, **a_observer_kwargs)

    @staticmethod
    def check_qconfig(qconfig: Union[Dict, Config]):
        """Check whether the passed qconfig's format meets requirement."""
        is_pass = True
        for arg in RequiredArgs:
            val = qconfig.get(arg, None)
            if isinstance(val, dict) and arg in qconfig.keys():
                continue
            else:
                is_pass = False
                break
        return is_pass

    def convert(self):
        """Generate torch's QConfig with built fake_quants."""
        torch_qconfig = QConfig(
            weight=self.w_fake_quant, activation=self.a_fake_quant)
        return torch_qconfig


class QSchemeHander(object):
    """Convert the qscheme of custom user-friendly qconfig to args needed in
    observers.

    Args:
        qdtype (str): Quantization dtype. It should is 'quint8' or 'qint8',
            and should be supported by the deploy backend. Defaults to 'quint8'
        bit (int): Quantization bit number. Defaults to 8.
        is_symmetry (bool): Is symmetry quantization or not. Defaults to True.
        is_per_channel (bool): Is per-channel quantization or not.
            Defaults to False.
    """

    def __init__(self,
                 qdtype: str = 'quint8',
                 bit: int = 8,
                 is_symmetry: bool = True,
                 is_per_channel: bool = False,
                 **kwargs):
        assert qdtype in ('quint8', 'qint8'), \
            'qdtype is incorrect, it should be quint8 or qint8.'
        self.qdtype = qdtype
        self.bit = bit
        self.is_symmetry = is_symmetry
        self.is_per_channel = is_per_channel

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
        """Generate the args needed in observers."""
        if self.qdtype == 'quint8':
            quant_min = 0
            quant_max = 2**self.bit - 1
        else:
            quant_max = 2**(self.bit - 1) - 1
            if self.is_symmetric_range:
                quant_min = -2**(self.bit - 1) + 1
            else:
                quant_min = -2**(self.bit - 1)

        # `dtype` will be same as BackenConfig's
        naive_para = {
            'dtype': torch.quint8 if self.qdtype == 'quint8' else torch.qint8,
            'quant_min': quant_min,
            'quant_max': quant_max,
            'qscheme': self.torch_qscheme,
            'reduce_range': False
        }
        if self.is_per_channel:
            naive_para['ch_axis'] = 0
        all_para = self.kwargs.copy()
        all_para.update(naive_para)
        return all_para

    def __str__(self):
        """Print generated args for observers."""
        return f'dtype: {self.dtype} / bit: {self.bit} / is_symmetry: {self.is_symmetry} / \
                is_per_channel: {self.is_per_channel} \
                / extra_kwargs: {self.kwargs}'


if __name__ == '__main__':
    from mmrazor.models.fake_quants import register_torch_fake_quants
    from mmrazor.models.observers import register_torch_observers
    register_torch_observers()
    register_torch_fake_quants()

    qconfig = dict(
        w_observer=dict(type='mmrazor.MovingAveragePerChannelMinMaxObserver'),
        a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
        w_fake_quant=dict(type='mmrazor.FakeQuantize'),
        a_fake_quant=dict(type='mmrazor.FakeQuantize'),
        w_qscheme=dict(
            qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
        a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
    )
    from mmengine.config import Config
    qconfig = Config(qconfig)
    torch_qconfig = QConfigHander(qconfig).convert()
    print(torch_qconfig)
