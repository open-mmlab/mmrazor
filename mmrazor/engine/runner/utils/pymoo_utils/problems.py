# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from pymoo.core.problem import Problem


class AuxiliarySingleLevelProblem(Problem):
    """The optimization problem for finding the next N candidate
    architectures."""

    def __init__(self, search_loop, dim=15, sec_obj='flops'):
        super().__init__(n_var=dim, n_obj=2, vtype=np.int32)

        self.search_loop = search_loop
        self.predictor = self.search_loop.predictor
        self.sec_obj = sec_obj

        self.xl = np.zeros(self.n_var)
        # upper bound for variable, automatically calculate by search space
        self.xu = []
        from mmrazor.models.mutables import OneShotMutableChannelUnit
        for mutable in self.predictor.search_groups.values():
            if isinstance(mutable[0], OneShotMutableChannelUnit):
                if mutable[0].num_channels > 0:
                    self.xu.append(mutable[0].num_channels - 1)
            else:
                if mutable[0].num_choices > 0:
                    self.xu.append(mutable[0].num_choices - 1)
        self.xu = np.array(self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate results."""
        f = np.full((x.shape[0], self.n_obj), np.nan)
        # predicted top1 error
        top1_err = self.predictor.handler.predict(x)[:, 0]
        for i in range(len(x)):
            candidate = self.predictor.vector2model(x[i])
            _, resource = self.search_loop._check_constraints(candidate)
            f[i, 0] = top1_err[i]
            f[i, 1] = resource[self.sec_obj]

        out['F'] = f


class SubsetProblem(Problem):
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
