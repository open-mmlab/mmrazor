# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Tuple

import torch

from mmrazor.models import ResourceEstimator
from mmrazor.structures import export_fix_subnet
from mmrazor.utils import SupportRandomSubnet

try:
    from mmdet.models.detectors import BaseDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDetector = get_placeholder('mmdet')


@torch.no_grad()
def check_subnet_resources(
    model,
    subnet: SupportRandomSubnet,
    estimator: ResourceEstimator,
    constraints_range: Dict[str, Any] = dict(flops=(0, 330))
) -> Tuple[bool, Dict]:
    """Check whether is beyond resources constraints.

    Returns:
        bool, result: The result of checking.
    """
    if constraints_range is None:
        return True, dict()

    assert hasattr(model, 'mutator') and hasattr(model, 'architecture')
    model.mutator.set_choices(subnet)
    _, sliced_model = export_fix_subnet(model, slice_weight=True)

    model_to_check = sliced_model.architecture  # type: ignore
    if isinstance(model_to_check, BaseDetector):
        results = estimator.estimate(model=model_to_check.backbone)
    else:
        results = estimator.estimate(model=model_to_check)

    for k, v in constraints_range.items():
        if not isinstance(v, (list, tuple)):
            v = (0, v)
        if results[k] < v[0] or results[k] > v[1]:
            return False, results

    return True, results
