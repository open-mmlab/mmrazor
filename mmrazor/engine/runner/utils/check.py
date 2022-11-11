# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple

import torch.nn as nn

from mmrazor.models.task_modules import ResourceEstimator
from mmrazor.structures import export_fix_subnet, load_fix_subnet
from mmrazor.utils import SupportRandomSubnet

try:
    from mmdet.models.detectors import BaseDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDetector = get_placeholder('mmdet')


def check_subnet_flops(
        model: nn.Module,
        subnet: SupportRandomSubnet,
        estimator: ResourceEstimator,
        flops_range: Optional[Tuple[float, float]] = None) -> bool:
    """Check whether is beyond flops constraints.

    Returns:
        bool: The result of checking.
    """
    if flops_range is None:
        return True

    assert hasattr(model, 'set_subnet') and hasattr(model, 'architecture')
    model.set_subnet(subnet)
    fix_mutable = export_fix_subnet(model)
    copied_model = copy.deepcopy(model)
    load_fix_subnet(copied_model, fix_mutable)

    model_to_check = model.architecture
    if isinstance(model_to_check, BaseDetector):
        results = estimator.estimate(model=model_to_check.backbone)
    else:
        results = estimator.estimate(model=model_to_check)

    flops = results['flops']
    flops_mix, flops_max = flops_range
    if flops_mix <= flops <= flops_max:  # type: ignore
        return True
    else:
        return False
