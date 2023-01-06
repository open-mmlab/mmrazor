# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.fake_quants import register_torch_fake_quants
from mmrazor.registry import MODELS


def test_register_torch_fake_quants():

    TORCH_fake_quants = register_torch_fake_quants()
    assert isinstance(TORCH_fake_quants, list)
    for fake_quant in TORCH_fake_quants:
        assert MODELS.get(fake_quant)
