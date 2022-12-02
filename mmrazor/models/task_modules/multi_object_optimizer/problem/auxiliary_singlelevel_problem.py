# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from pymoo.core.problem import Problem as BaseProblem


class AuxiliarySingleLevelProblem(BaseProblem):
    """The optimization problem for finding the next N candidate
    architectures."""

    def __init__(self, searcher, dim=15, sec_obj='flops'):
        super().__init__(n_var=dim, n_obj=2, vtype=np.int32)

        self.searcher = searcher
        self.predictor = self.searcher.predictor
        self.sec_obj = sec_obj

        self.xl = np.zeros(self.n_var)
        # upper bound for variable, automatically calculate by search space
        self.xu = []
        for mutable in self.predictor.search_groups.values():
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
            _, resource = self.searcher._check_constraints(candidate)
            f[i, 0] = top1_err[i]
            f[i, 1] = resource[self.sec_obj]

        out['F'] = f
