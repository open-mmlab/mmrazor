# Copyright (c) OpenMMLab. All rights reserved.
import pytest


def test_v0_5_deprecated_set_random_seed() -> None:
    warn_msg = 'Deprecated in v0.3 and will be removed in v0.5, ' \
        'please import `set_random_seed` directly from `mmrazor.apis`'
    from mmrazor.apis.mmcls import set_random_seed
    with pytest.deprecated_call(match=warn_msg):
        set_random_seed(123)

    from mmrazor.apis.mmdet import set_random_seed
    with pytest.deprecated_call(match=warn_msg):
        set_random_seed(123)

    from mmrazor.apis.mmseg import set_random_seed
    with pytest.deprecated_call(match=warn_msg):
        set_random_seed(123)
