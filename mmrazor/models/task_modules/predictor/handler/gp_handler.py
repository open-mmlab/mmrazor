# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from pydacefit.corr import (corr_cubic, corr_exp, corr_expg, corr_gauss,
                            corr_spherical, corr_spline)
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.fit import fit as pydace_fit
from pydacefit.regr import regr_constant

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler

REGR = {
    'linear': regr_linear,
    'constant': regr_constant,
    'quadratic': regr_quadratic
}

CORR = {
    'gauss': corr_gauss,
    'cubic': corr_cubic,
    'exp': corr_exp,
    'expg': corr_expg,
    'spline': corr_spline,
    'spherical': corr_spherical
}


class DACE_with_smooth(DACE):
    """GP model."""

    def __init__(self,
                 regr,
                 corr,
                 theta: float = 1.0,
                 thetaL: float = 0.0,
                 thetaU: float = 100.0):
        super(DACE_with_smooth, self).__init__(regr, corr, theta, thetaL,
                                               thetaU)

    def fit(self, X, Y):
        """Build the model."""
        if len(Y.shape) == 1:
            Y = Y[:, None]

        if X.shape[0] != Y.shape[0]:
            raise Exception('X and Y must have the same number of rows.')

        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1) + 1e-6
        mY, sY = np.mean(Y, axis=0), np.std(Y, axis=0, ddof=1) + 1e-6

        nX = (X - mX) / sX
        nY = (Y - mY) / sY

        if self.tl is not None and self.tu is not None:
            self.model = {'nX': nX, 'nY': nY}
            self.boxmin()
            self.model = self.itpar['best']
        else:
            self.model = pydace_fit(nX, nY, self.regr, self.kernel, self.theta)

        self.model = {
            **self.model, 'mX': mX,
            'sX': sX,
            'mY': mY,
            'sY': sY,
            'nX': nX,
            'nY': nY
        }
        self.model['sigma2'] = np.square(sY) @ self.model['_sigma2']


@TASK_UTILS.register_module()
class GaussProcessHandler(BaseHandler):
    """GaussProcess handler of the metric predictor. It uses Gaussian Process
    (Kriging) to predict the metric of a trained model.

    Args:
        regr (str): regression kernel for GP model. Defaults to 'linear'.
        corr (str): correlation kernel for GP model. Defaults to 'gauss'.
    """

    def __init__(self, regr: str = 'linear', corr: str = 'gauss'):
        assert regr in REGR, \
            ValueError(f'`regr` should be in `REGR`. Got `{regr}`.')
        assert corr in CORR, \
            ValueError(f'`corr` should be in `CORR`. Got `{corr}`.')
        self.regr = REGR[regr]
        self.corr = CORR[corr]

        self.model = DACE_with_smooth(
            regr=self.regr,
            corr=self.corr,
            theta=1.0,
            thetaL=0.00001,
            thetaU=100)

    def fit(self, train_data: np.array, train_label: np.array) -> None:
        """Training the model of handler.

        Args:
            train_data (numpy.array): input data for training.
            train_label (numpy.array): input label for training.
        """
        self.model.fit(train_data, train_label)

    def predict(self, test_data: np.array) -> np.array:
        """Predict the evaluation metric of the model.

        Args:
            test_data (numpy.array): input data for testing.

        Returns:
            numpy.array: predicted metric.
        """
        return self.model.predict(test_data)
