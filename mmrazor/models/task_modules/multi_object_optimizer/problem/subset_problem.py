# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .base_problem import BaseProblem


class SubsetProblem(BaseProblem):
    """select a subset to diversify the pareto front."""

    def __init__(self, candidates, archive, K):
        super().__init__(
            n_var=len(candidates),
            n_obj=1,
            n_constr=1,
            xl=0,
            xu=1,
            type_var=bool)
        self.archive = archive
        self.candidates = candidates
        self.n_max = K

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], 1), np.nan)
        g = np.full((x.shape[0], 1), np.nan)

        for i, _x in enumerate(x):
            # append selected candidates to archive then sort
            tmp = np.sort(np.concatenate((self.archive, self.candidates[_x])))
            f[i, 0] = np.std(np.diff(tmp))
            # we penalize if the number of selected candidates is not exactly K
            g[i, 0] = (self.n_max - np.sum(_x))**2

        out['F'] = f
        out['G'] = g
