# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrazor import digit_version
from mmrazor.models.observers import register_torch_observers
from mmrazor.registry import MODELS


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.13.0'),
    reason='version of torch < 1.13.0')
def test_register_torch_observers():

    TORCH_observers = register_torch_observers()
    assert isinstance(TORCH_observers, list)
    for observer in TORCH_observers:
        assert MODELS.get(observer)
