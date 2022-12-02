# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np

from .domin_matrix import get_relation
# from pymoo.core.population import Population
# from pymoo.util.misc import crossover_mask, random_permuations
# from pymoo.operators.repair.bounce_back import bounce_back_by_problem
from .helper import (Population, crossover_mask, random_permuations,
                     repair_out_of_bounds)


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError('Only implemented for binary tournament!')

    S = np.full(P.shape[0], np.nan)

    for Index in range(P.shape[0]):

        a, b = P[Index, 0], P[Index, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[Index] = compare(
                a,
                pop[a].CV,
                b,
                pop[b].CV,
                method='smaller_is_better',
                return_random_if_equal=True)
        else:
            rel = get_relation(pop[a].F, pop[b].F)
            if rel == 1:
                S[Index] = a
            elif rel == -1:
                S[Index] = b
            # if rank or domination relation didn't make a decision
            if np.isnan(S[Index]):
                S[Index] = compare(
                    a,
                    pop[a].get('crowding'),
                    b,
                    pop[b].get('crowding'),
                    method='larger_is_better',
                    return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for Index in range(P.shape[0]):
        a, b = P[Index, 0], P[Index, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[Index] = compare(
                a,
                pop[a].CV,
                b,
                pop[b].CV,
                method='smaller_is_better',
                return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[Index] = compare(
                a,
                pop[a].F,
                b,
                pop[b].F,
                method='smaller_is_better',
                return_random_if_equal=True)

    return S[:, None].astype(int)


def compare(a, a_val, b, b_val, method, return_random_if_equal=False):
    if method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None
    elif method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            if return_random_if_equal:
                return np.random.choice([a, b])
            else:
                return None


class TournamentSelection:
    """The Tournament selection is used to simulated a tournament between
    individuals.

    The pressure balances greedy the genetic algorithm will be.
    """

    def __init__(self, pressure=2, func_comp='binary_tournament'):
        """

        Parameters
        ----------
        func_comp: func
            The function to compare two individuals.
            It has the shape: comp(pop, indices) and returns the winner.

        pressure: int
            The selection pressure to bie applied.
        """

        # selection pressure to be applied
        self.pressure = pressure
        if func_comp == 'comp_by_cv_and_fitness':
            self.f_comp = comp_by_cv_and_fitness
        else:
            self.f_comp = binary_tournament

    def do(self, pop, n_select, n_parents=2, **kwargs):
        # number of random individuals needed
        n_random = n_select * n_parents * self.pressure

        # number of permutations needed
        n_perms = math.ceil(n_random / len(pop))

        # get random permutations and reshape them
        P = random_permuations(n_perms, len(pop))[:n_random]
        P = np.reshape(P, (n_select * n_parents, self.pressure))

        # compare using tournament function
        S = self.f_comp(pop, P, **kwargs)

        return np.reshape(S, (n_select, n_parents))


class PointCrossover:

    def __init__(self, n_points=2, n_parents=2, n_offsprings=2, prob=0.9):
        self.n_points = n_points
        self.prob = prob
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

    def do(self, problem, pop, parents, **kwargs):
        """

        Parameters
        ----------
        problem: class
            The problem to be solved.

        pop : Population
            The population as an object

        parents: numpy.array
            The select parents of the population for the crossover

        kwargs : dict
            Any additional data that might be necessary.

        Returns
        -------
        offsprings : Population
            The off as a matrix. n_children rows and the number of columns is
            equal to the variable length of the problem.

        """

        if self.n_parents != parents.shape[1]:
            raise ValueError(
                'Exception during crossover: '
                'Number of parents differs from defined at crossover.')

        # get the design space matrix form the population and parents
        X = pop.get('X')[parents.T].copy()

        # now apply the crossover probability
        do_crossover = np.random.random(len(parents)) < self.prob

        # execute the crossover
        _X = self._do(problem, X, **kwargs)

        X[:, do_crossover, :] = _X[:, do_crossover, :]

        # flatten the array to become a 2d-array
        X = X.reshape(-1, X.shape[-1])

        # create a population object
        off = pop.new('X', X)

        return off

    def _do(self, problem, X, **kwargs):

        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # start point of crossover
        r = np.row_stack([
            np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)
        ])[:, :self.n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, n_var)])

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # create for each individual the crossover range
        for Index in range(n_matings):

            j = 0
            while j < r.shape[1] - 1:
                a, b = r[Index, j], r[Index, j + 1]
                M[Index, a:b] = True
                j += 2

        _X = crossover_mask(X, M)

        return _X


class PolynomialMutation:

    def __init__(self, eta=20, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, X, **kwargs):

        Y = np.full(X.shape, np.inf)

        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        do_mutation = np.random.random(X.shape) < self.prob

        Y[:, :] = X

        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (
            np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (
            np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        # in case out of bounds repair (very unlikely)
        # Y = bounce_back_by_problem(problem, Y)
        Y = repair_out_of_bounds(problem, Y)

        return Y

    def do(self, problem, pop, **kwargs):
        Y = self._do(problem, pop.get('X'), **kwargs)
        return pop.new('X', Y)


class IntegerFromFloatMutation:

    def __init__(self, **kwargs):

        self.mutation = PolynomialMutation(**kwargs)

    def _do(self, problem, X, **kwargs):

        def fun():
            return self.mutation._do(problem, X, **kwargs)

        # save the original bounds of the problem
        _xl, _xu = problem.xl, problem.xu

        # copy the arrays of the problem and cast them to float
        xl, xu = problem.xl, problem.xu

        # modify the bounds to match the new crossover specifications
        problem.xl = xl - (0.5 - 1e-16)
        problem.xu = xu + (0.5 - 1e-16)

        # perform the crossover
        off = fun()

        # now round to nearest integer for all offsprings
        off = np.rint(off)

        # reset the original bounds of the problem and design space values
        problem.xl = _xl
        problem.xu = _xu

        return off

    def do(self, problem, pop, **kwargs):
        """Mutate variables in a genetic way.

        Parameters
        ----------
        problem : class
            The problem instance
        pop : Population
            A population object

        Returns
        -------
        Y : Population
            The mutated population.
        """

        Y = self._do(problem, pop.get('X'), **kwargs)
        return pop.new('X', Y)


class Mating:

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 repair=None,
                 eliminate_duplicates=None,
                 n_max_iterations=100):

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.n_max_iterations = n_max_iterations
        self.eliminate_duplicates = eliminate_duplicates
        self.repair = repair

    def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):

        # if the parents for the mating are not provided directly
        if parents is None:

            # how many parents need to be select for the mating
            n_select = math.ceil(n_offsprings / self.crossover.n_offsprings)

            # select the parents for the mating - just an index array
            parents = self.selection.do(pop, n_select,
                                        self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population
        _off = self.crossover.do(problem, pop, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        _off = self.mutation.do(problem, _off, **kwargs)

        return _off

    def do(self, problem, pop, n_offsprings, **kwargs):

        # the population object to be used
        off = pop.new()

        # infill counter
        # counts how often the mating needs to be done to fill up n_offsprings
        n_infills = 0
        # iterate until enough offsprings are created
        while len(off) < n_offsprings:

            # how many offsprings are remaining to be created
            n_remaining = n_offsprings - len(off)

            # do the mating
            _off = self._do(problem, pop, n_remaining, **kwargs)

            # repair the individuals if necessary
            if self.repair:
                _off = self.repair.do(problem, _off, **kwargs)

            if self.eliminate_duplicates is not None:
                _off = self.eliminate_duplicates.do(_off, pop, off)

            # if more offsprings than necessary - truncate them randomly
            if len(off) + len(_off) > n_offsprings:
                n_remaining = n_offsprings - len(off)
                _off = _off[:n_remaining]

            # add to the offsprings and increase the mating counter
            off = off.merge(_off)
            n_infills += 1

            if n_infills > self.n_max_iterations:
                break

        return off


class MySampling:

    def __init__(self):
        pass

    def do(self, problem, n_samples, pop=Population(), **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            Index = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, Index] = True

        if pop is None:
            return X
        return pop.new('X', X)


class BinaryCrossover(PointCrossover):

    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            Index = np.where(np.logical_xor(p1, p2))[0]

            S = Index[np.random.permutation(len(Index))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(PolynomialMutation):

    def _do(self, problem, X, **kwargs):
        for Index in range(X.shape[0]):
            X[Index, :] = X[Index, :]
            is_false = np.where(np.logical_not(X[Index, :]))[0]
            is_true = np.where(X[Index, :])[0]
            try:
                X[Index, np.random.choice(is_false)] = True
                X[Index, np.random.choice(is_true)] = False
            except ValueError:
                pass

        return X
