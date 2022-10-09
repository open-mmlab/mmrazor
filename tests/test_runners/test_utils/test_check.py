# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

from mmrazor.engine.runner.utils import check_subnet_resources

try:
    from mmdet.models.detectors import BaseDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDetector = get_placeholder('mmdet')


@patch('mmrazor.models.ResourceEstimator')
@patch('mmrazor.models.SPOS')
def test_check_subnet_resources(mock_model, mock_estimator):
    # constraints_range = dict()
    constraints_range = dict()
    fake_subnet = {'1': 'choice1', '2': 'choice2'}
    is_pass, _ = check_subnet_resources(mock_model, fake_subnet,
                                        mock_estimator, constraints_range)
    assert is_pass is True

    # constraints_range is not None
    # architecturte is BaseDetector
    constraints_range = dict(flops=(0, 330))
    mock_model.architecture = BaseDetector
    fake_results = {'flops': 50.}
    mock_estimator.estimate.return_value = fake_results
    is_pass, _ = check_subnet_resources(
        mock_model,
        fake_subnet,
        mock_estimator,
        constraints_range,
    )
    assert is_pass is True

    # constraints_range is not None
    # architecturte is BaseDetector
    constraints_range = dict(flops=(0, 330))
    fake_results = {'flops': -50.}
    mock_estimator.estimate.return_value = fake_results
    is_pass, _ = check_subnet_resources(mock_model, fake_subnet,
                                        mock_estimator, constraints_range)
    assert is_pass is False
