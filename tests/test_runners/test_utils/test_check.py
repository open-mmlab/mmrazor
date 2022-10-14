# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

from mmrazor.engine.runner.utils import check_subnet_flops

try:
    from mmdet.models.detectors import BaseDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDetector = get_placeholder('mmdet')


@patch('mmrazor.models.ResourceEstimator')
@patch('mmrazor.models.SPOS')
def test_check_subnet_flops(mock_model, mock_estimator):
    # flops_range = None
    flops_range = None
    fake_subnet = {'1': 'choice1', '2': 'choice2'}
    result = check_subnet_flops(mock_model, fake_subnet, mock_estimator,
                                flops_range)
    assert result is True

    # flops_range is not None
    # architecturte is BaseDetector
    flops_range = (0., 100.)
    mock_model.architecture = BaseDetector
    fake_results = {'flops': 50.}
    mock_estimator.estimate.return_value = fake_results
    result = check_subnet_flops(mock_model, fake_subnet, mock_estimator,
                                flops_range)
    assert result is True

    # flops_range is not None
    # architecturte is BaseDetector
    flops_range = (0., 100.)
    fake_results = {'flops': -50.}
    mock_estimator.estimate.return_value = fake_results
    result = check_subnet_flops(mock_model, fake_subnet, mock_estimator,
                                flops_range)
    assert result is False
