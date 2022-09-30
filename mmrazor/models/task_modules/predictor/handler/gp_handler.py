# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

try:
    import pydacefit
    from pydacefit.dace import DACE
except ImportError:
    pydacefit = None
    DACE = object

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


def get_func():
    if pydacefit is None:
        raise RuntimeError('Failed to import pydacefit. Please run '
                           '"pip install pydacefit". ')

    from pydacefit.corr import (corr_cubic, corr_exp, corr_expg, corr_gauss,
                                corr_spherical, corr_spline)
    from pydacefit.dace import regr_linear, regr_quadratic
    from pydacefit.regr import regr_constant

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

    return REGR, CORR


class DACE_with_smooth(DACE):
    """GP model."""

    def __init__(self, regr, corr, theta=1, thetaL=0, thetaU=100):
        super(DACE_with_smooth, self).__init__(regr, corr, theta, thetaL,
                                               thetaU)

    def fit(self, X, Y):

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
            from pydacefit.fit import fit
            self.model = fit(nX, nY, self.regr, self.kernel, self.theta)

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
    """Gaussian Process (Kriging)

    Args:
        regr (str): regression kernel for GP model.
        corr (str): correlation kernel for GP model.
    """

    def __init__(self, regr='linear', corr='gauss'):
        REGR, CORR = get_func()
        assert regr in REGR and corr in CORR, \
            NotImplementedError('Unknown GP regression or correlation !')
        self.regr = REGR[regr]
        self.corr = CORR[corr]

        self.model = DACE_with_smooth(
            regr=self.regr,
            corr=self.corr,
            theta=1.0,
            thetaL=0.00001,
            thetaU=100)

    def fit(self, train_data, train_label):
        """Training predictor."""
        self.model.fit(train_data, train_label)

    def predict(self, test_data):
        """Predict the subnets' performance."""
        return self.model.predict(test_data)
