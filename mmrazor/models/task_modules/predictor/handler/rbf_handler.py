# Copyright (c) OpenMMLab. All rights reserved.
try:
    import pySOT
except ImportError:
    pySOT = None

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


@TASK_UTILS.register_module()
class RBFHandler(BaseHandler):
    """Radial Basis Function.

    Args:
        kernel (str): RBF kernel object.
        tail (str): RBF polynomial tail object.
    """

    def __init__(self, kernel='tps', tail='linear'):
        if pySOT is None:
            raise RuntimeError('Failed to import pydacefit. Please run '
                               '"pip install pySOT==0.2.3". ')
        from pySOT.surrogate import (ConstantTail, CubicKernel, LinearTail,
                                     TPSKernel)
        self.kernel = kernel
        self.tail = tail
        self.model = None

        if kernel == 'cubic':
            self.kernel = CubicKernel
        elif self.kernel == 'tps':
            self.kernel = TPSKernel
        else:
            raise NotImplementedError('unknown RBF kernel')

        if tail == 'linear':
            self.tail = LinearTail
        elif self.tail == 'constant':
            self.tail = ConstantTail
        else:
            raise NotImplementedError('unknown RBF tail')

    def fit(self, train_data, train_label):
        """Training predictor."""
        if train_data.shape[0] <= train_data.shape[1]:
            raise ValueError('RBF only support '
                             f'# of samples{train_data.shape[0]}'
                             f' > # of dimensions{train_data.shape[1]} !')
        from pySOT.surrogate import RBFInterpolant
        self.model = RBFInterpolant(
            dim=train_data.shape[1],
            kernel=self.kernel(),
            tail=self.tail(train_data.shape[1]))

        for i in range(len(train_data)):
            self.model.add_points(train_data[i, :], train_label[i])

    def predict(self, test_data):
        """Predict the subnets' performance."""
        return self.model.predict(test_data)
