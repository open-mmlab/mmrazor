# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from pySOT.surrogate import (ConstantTail, CubicKernel, Kernel, LinearTail,
                             RBFInterpolant, Tail, TPSKernel)

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


@TASK_UTILS.register_module()
class RBFHandler(BaseHandler):
    """RBF handler of the metric predictor. It uses `Radial Basis Function` to
    predict the metric of a trained model.

    Args:
        kernel (str): RBF kernel object. Defaults to 'tps'.
        tail (str): RBF polynomial tail object. Defaults to 'linear'.
    """
    kernel_mapping = {'cubic': CubicKernel, 'tps': TPSKernel}
    tail_mapping = {'linear': LinearTail, 'constant': ConstantTail}

    def __init__(self, kernel: str = 'tps', tail: str = 'linear'):
        assert kernel in self.kernel_mapping.keys(), (
            f'Got unknown RBF kernel `{kernel}`.')
        self.kernel: Kernel = self.kernel_mapping[kernel]

        assert tail in self.tail_mapping.keys(), (
            f'Got unknown RBF tail `{tail}`.')
        self.tail: Tail = self.tail_mapping[tail]

    def fit(self, train_data: np.array, train_label: np.array) -> None:
        """Training the model of handler.

        Args:
            train_data (numpy.array): input data for training.
            train_label (numpy.array): input label for training.
        """
        if train_data.shape[0] <= train_data.shape[1]:
            raise ValueError('In RBF, dim 0 of data (got '
                             f'{train_data.shape[0]}) should be larger than '
                             f'dim 1 of data (got {train_data.shape[1]}).')

        self.model = RBFInterpolant(
            dim=train_data.shape[1],
            kernel=self.kernel(),
            tail=self.tail(train_data.shape[1]))

        for i in range(len(train_data)):
            self.model.add_points(train_data[i, :], train_label[i])

    def predict(self, test_data: np.array) -> np.array:
        """Predict the evaluation metric of the model.

        Args:
            test_data (numpy.array): input data for testing.

        Returns:
            numpy.array: predicted metric.
        """
        return self.model.predict(test_data)
