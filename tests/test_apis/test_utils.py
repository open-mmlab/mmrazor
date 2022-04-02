# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmrazor.apis import set_random_seed


def test_set_random_seed() -> None:
    set_random_seed(123, False)
    x1 = torch.rand(3, 3)
    x2 = np.random.rand(3, 3)

    set_random_seed(123, True)
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    y1 = torch.rand(3, 3)
    y2 = np.random.rand(3, 3)

    assert torch.allclose(x1, y1, 1e-6)
    assert np.allclose(x2, y2, 1e-6)
