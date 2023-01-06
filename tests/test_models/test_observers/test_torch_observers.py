# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.observers import register_torch_observers
from mmrazor.registry import MODELS


def test_register_torch_observers():

    TORCH_observers = register_torch_observers()
    assert isinstance(TORCH_observers, list)
    for observer in TORCH_observers:
        assert MODELS.get(observer)
