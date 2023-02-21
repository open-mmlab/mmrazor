# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

try:
    from pymoo.model.crossover import Crossover
    from pymoo.model.mutation import Mutation
    from pymoo.model.problem import Problem
    from pymoo.model.sampling import Sampling
except ImportError:
    from mmrazor.utils import get_placeholder
    Crossover = get_placeholder('pymoo')
    Mutation = get_placeholder('pymoo')
    Problem = get_placeholder('pymoo')
    Sampling = get_placeholder('pymoo')


class RandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            index = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, index] = True

        return X


class BinaryCrossover(Crossover):

    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            index = np.where(np.logical_xor(p1, p2))[0]

            S = index[np.random.permutation(len(index))][:n_remaining]
            _X[0, k, S] = True

        return _X


class RandomMutation(Mutation):

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            try:
                X[i, np.random.choice(is_false)] = True
                X[i, np.random.choice(is_true)] = False
            except ValueError:
                pass

        return X


class AuxiliarySingleLevelProblem(Problem):
    """The optimization problem for finding the next N candidate
    architectures."""

    def __init__(self, search_loop, n_var=15, sec_obj='flops'):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, type_var=np.int)

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
