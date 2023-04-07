# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import torch
from mmengine.config import Config

try:
    from torch.ao.quantization import FakeQuantize, QConfig
    from torch.ao.quantization.utils import is_per_tensor
except ImportError:
    from mmrazor.utils import get_placeholder
    QConfig = get_placeholder('torch>=1.13')
    FakeQuantize = get_placeholder('torch>=1.13')
    is_per_tensor = get_placeholder('torch>=1.13')

from mmrazor.registry import MODELS

RequiredArgs = [
    'w_qscheme', 'a_qscheme', 'w_fake_quant', 'a_fake_quant', 'w_observer',
    'a_observer'
]

RetainArgsPerTensor = [
    'dtype', 'qscheme', 'quant_min', 'quant_max', 'reduce_range'
]
RetainArgsPerChannel = RetainArgsPerTensor + ['ch_axis']


class QSchemeHandler(object):
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


class QConfigHandler():
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
            self.w_qscheme = QSchemeHandler(
                is_per_channel=w_is_per_channel, **qconfig['w_qscheme'])
            self.a_qscheme = QSchemeHandler(
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

    @staticmethod
    def replace_fakequant(fake_quant_org: FakeQuantize,
                          qscheme_org: QSchemeHandler,
                          update_qparams: bool = True):
        """Replace origin fakequants in model with the specified fakequant,
        which is in favor of deploying the quantized model."""
        assert isinstance(qscheme_org, QSchemeHandler)
        observer_kwargs = qscheme_org.to_observer_params()
        if is_per_tensor(observer_kwargs['qscheme']):
            observer = MODELS.get('MinMaxObserver')
            retain_args = RetainArgsPerTensor
        else:
            observer = MODELS.get('PerChannelMinMaxObserver')
            retain_args = RetainArgsPerChannel
        pop_keys = []
        for k in observer_kwargs.keys():
            if k not in retain_args:
                pop_keys.append(k)
        for k in pop_keys:
            observer_kwargs.pop(k)
        fake_quant = MODELS.get('FakeQuantize')
        fake_quant_wrapper = fake_quant.with_args(
            observer=observer, **observer_kwargs)
        if update_qparams:
            device = fake_quant_org.scale.device
            fake_quant_ins = fake_quant_wrapper().to(device)
            fake_quant_ins.scale.copy_(fake_quant_org.scale)
            fake_quant_ins.zero_point.copy_(fake_quant_org.zero_point)
            obs = fake_quant_ins.activation_post_process
            obs_org = fake_quant_org.activation_post_process
            obs.min_val.resize_(obs_org.min_val.shape).copy_(obs_org.min_val)
            obs.max_val.resize_(obs_org.max_val.shape).copy_(obs_org.max_val)
            return fake_quant_ins
        else:
            return fake_quant_wrapper

    def fixed_w_fakequant(self):
        """Make `self.w_fake_quant` fixed as the consistent fakequant."""
        self.w_fake_quant = self.replace_fakequant(
            self.w_fake_quant(), self.w_qscheme, update_qparams=False)
