# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrazor import digit_version
from mmrazor.models.fake_quants import register_torch_fake_quants
from mmrazor.registry import MODELS


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    reason='version of torch < 1.13.0')
def test_register_torch_fake_quants():

    TORCH_fake_quants = register_torch_fake_quants()
    assert isinstance(TORCH_fake_quants, list)
    for fake_quant in TORCH_fake_quants:
        assert MODELS.get(fake_quant)
