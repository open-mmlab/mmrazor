# Copyright (c) OpenMMLab. All rights reserved.
# copied from https://github.com/anyoptimization/pymoo
import copy

import numpy as np
import scipy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def default_attr(pop):
    return pop.get('X')


def cdist(x, y):
    return scipy.spatial.distance.cdist(x, y)


class DuplicateElimination:
    """Implementation of Elimination.

    func: function to execute.
    """

    def __init__(self, func=None) -> None:
        super().__init__()
        self.func = func

        if self.func is None:
            self.func = default_attr

    def do(self, pop, *args, return_indices=False, to_itself=True):
        original = pop

        if to_itself:
            pop = pop[~self._do(pop, None, np.full(len(pop), False))]

        for arg in args:
            if len(arg) > 0:

                if len(pop) == 0:
                    break
                elif len(arg) == 0:
                    continue
                else:
                    pop = pop[~self._do(pop, arg, np.full(len(pop), False))]

        if return_indices:
            no_duplicate, is_duplicate = [], []
            H = set(pop)

            for Index, ind in enumerate(original):
                if ind in H:
                    no_duplicate.append(Index)
                else:
                    is_duplicate.append(Index)

            return pop, no_duplicate, is_duplicate
        else:
            return pop


class DefaultDuplicateElimination(DuplicateElimination):
    """Implementation of DefaultDuplicate Elimination.

    epsilon(float): smallest dist for judge duplication.
    """

    def __init__(self, epsilon=1e-16, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def calc_dist(self, pop, other=None):
        X = self.func(pop)

        if other is None:
            D = cdist(X, X)
            D[np.triu_indices(len(X))] = np.inf
        else:
            _X = self.func(other)
            D = cdist(X, _X)

        return D

    def _do(self, pop, other, is_duplicate):
        D = self.calc_dist(pop, other)
        D[np.isnan(D)] = np.inf

        is_duplicate[np.any(D < self.epsilon, axis=1)] = True
        return is_duplicate


class Individual:
    """Class for each individual in search step."""

    def __init__(self,
                 X=None,
                 F=None,
                 CV=None,
                 G=None,
                 feasible=None,
                 **kwargs) -> None:
        self.X = X
        self.F = F
        self.CV = CV
        self.G = G
        self.feasible = feasible
        self.data = kwargs
        self.attr = set(self.__dict__.keys())

    def has(self, key):
        return key in self.attr or key in self.data

    def set(self, key, value):
        if key in self.attr:
            self.__dict__[key] = value
        else:
            self.data[key] = value

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, keys):
        if keys in self.data:
            return self.data[keys]
        elif keys in self.attr:
            return self.__dict__[keys]
        else:
            return None


class Population(np.ndarray):
    """Class for all the population in search step."""

    def __new__(cls, n_individuals=0, individual=Individual()):
        obj = super(Population, cls).__new__(
            cls, n_individuals, dtype=individual.__class__).view(cls)
        for Index in range(n_individuals):
            obj[Index] = individual.copy()
        obj.individual = individual
        return obj

    def merge(self, a, b=None):
        if b:
            a, b = pop_from_array_or_individual(a), \
                        pop_from_array_or_individual(b)
            a.merge(b)
        else:
            other = pop_from_array_or_individual(a)
            if len(self) == 0:
                return other
            else:
                obj = np.concatenate([self, other]).view(Population)
                obj.individual = self.individual
                return obj

    def copy(self):
        pop = Population(n_individuals=len(self), individual=self.individual)
        for Index in range(len(self)):
            pop[Index] = self[Index]
        return pop

    def has(self, key):
        return all([ind.has(key) for ind in self])

    def __deepcopy__(self, memo):
        return self.copy()

    @classmethod
    def create(cls, *args):
        pop = np.concatenate([
            pop_from_array_or_individual(arg) for arg in args
        ]).view(Population)
        pop.individual = Individual()
        return pop

    def new(self, *args):

        if len(args) == 1:
            return Population(
                n_individuals=args[0], individual=self.individual)
        else:
            n = len(args[1]) if len(args) > 0 else 0
            pop = Population(n_individuals=n, individual=self.individual)
            if len(args) > 0:
                pop.set(*args)
            return pop

    def collect(self, func, to_numpy=True):
        val = []
        for Index in range(len(self)):
            val.append(func(self[Index]))
        if to_numpy:
            val = np.array(val)
        return val

    def set(self, *args):

        for Index in range(int(len(args) / 2)):

            key, values = args[Index * 2], args[Index * 2 + 1]
            is_iterable = hasattr(values,
                                  '__len__') and not isinstance(values, str)

            if is_iterable and len(values) != len(self):
                raise Exception(
                    'Population Set Attribute Error: '
                    'Number of values and population size do not match!')

            for Index in range(len(self)):
                val = values[Index] if is_iterable else values
                self[Index].set(key, val)

        return self

    def get(self, *args, to_numpy=True):

        val = {}
        for c in args:
            val[c] = []

        for Index in range(len(self)):

            for c in args:
                val[c].append(self[Index].get(c))

        res = [val[c] for c in args]
        if to_numpy:
            res = [np.array(e) for e in res]

        if len(args) == 1:
            return res[0]
        else:
            return tuple(res)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.individual = getattr(obj, 'individual', None)


def pop_from_array_or_individual(array, pop=None):
    # the population type can be different
    if pop is None:
        pop = Population()

    # provide a whole population object
    if isinstance(array, Population):
        pop = array
    elif isinstance(array, np.ndarray):
        pop = pop.new('X', np.atleast_2d(array))
    elif isinstance(array, Individual):
        pop = Population(1)
        pop[0] = array
    else:
        return None

    return pop


class Initialization:
    """Initiallize step."""

    def __init__(self,
                 sampling,
                 individual=Individual(),
                 repair=None,
                 eliminate_duplicates=None) -> None:

        super().__init__()
        self.sampling = sampling
        self.individual = individual
        self.repair = repair
        self.eliminate_duplicates = eliminate_duplicates

    def do(self, problem, n_samples, **kwargs):

        # provide a whole population object
        if isinstance(self.sampling, Population):
            pop = self.sampling

        else:
            pop = Population(0, individual=self.individual)
            if isinstance(self.sampling, np.ndarray):
                pop = pop.new('X', self.sampling)
            else:
                pop = self.sampling.do(problem, n_samples, pop=pop, **kwargs)

        # repair all solutions that are not already evaluated
        if self.repair:
            Index = [k for k in range(len(pop)) if pop[k].F is None]
            pop = self.repair.do(problem, pop[Index], **kwargs)

        if self.eliminate_duplicates is not None:
            pop = self.eliminate_duplicates.do(pop)

        return pop


def split_by_feasibility(pop, sort_infeasbible_by_cv=True):
    CV = pop.get('CV')

    b = (CV <= 0)

    feasible = np.where(b)[0]
    infeasible = np.where(np.logical_not(b))[0]

    if sort_infeasbible_by_cv:
        infeasible = infeasible[np.argsort(CV[infeasible, 0])]

    return feasible, infeasible


class Survival:
    """The survival process is implemented inheriting from this class, which
    selects from a population only specific individuals to survive.

    Parameters
    ----------
    filter_infeasible : bool
        Whether for the survival infeasible solutions should be
        filtered first
    """

    def __init__(self, filter_infeasible=True):
        self.filter_infeasible = filter_infeasible

    def do(self, problem, pop, n_survive, return_indices=False, **kwargs):

        # if the split should be done beforehand
        if self.filter_infeasible and problem.n_constr > 0:
            feasible, infeasible = split_by_feasibility(
                pop, sort_infeasbible_by_cv=True)

            # if there was no feasible solution was added at all
            if len(feasible) == 0:
                survivors = pop[infeasible[:n_survive]]

            # if there are feasible solutions in the population
            else:
                survivors = pop.new()

                # if feasible solution do exist
                if len(feasible) > 0:
                    survivors = self._do(problem, pop[feasible],
                                         min(len(feasible), n_survive),
                                         **kwargs)

                # if infeasible solutions needs to be added
                if len(survivors) < n_survive:
                    least_infeasible = infeasible[:n_survive - len(feasible)]
                    survivors = survivors.merge(pop[least_infeasible])

        else:
            survivors = self._do(problem, pop, n_survive, **kwargs)

        if return_indices:
            H = {}
            for k, ind in enumerate(pop):
                H[ind] = k
            return [H[survivor] for survivor in survivors]
        else:
            return survivors

    def _do(self, problem, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get('F').astype(np.float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])
            # save rank and crowding in the individual class
            for j, Index in enumerate(front):
                pop[Index].set('rank', k)
                pop[Index].set('crowding', crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                Index = randomized_argsort(
                    crowding_of_front, order='descending', method='numpy')
                Index = Index[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                Index = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[Index])

        return pop[survivors]


class FitnessSurvival(Survival):
    """Survival class for Fitness."""

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, problem, pop, n_survive, out=None, **kwargs):
        F = pop.get('F')

        if F.shape[1] != 1:
            raise ValueError(
                'FitnessSurvival can only used for single objective single!')

        return pop[np.argsort(F[:, 0])[:n_survive]]


def find_duplicates(X, epsilon=1e-16):
    # calculate the distance matrix from each point to another
    D = cdist(X, X)

    # set the diagonal to infinity
    D[np.triu_indices(len(X))] = np.inf

    # set as duplicate if a point is really close to this one
    is_duplicate = np.any(D < epsilon, axis=1)

    return is_duplicate


def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates
            is_unique = np.where(
                np.logical_not(find_duplicates(F, epsilon=1e-24)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        Index = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[Index, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack(
            [np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[
            1:] / norm

        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives
        J = np.argsort(Index, axis=0)
        _cd = np.sum(
            dist_to_last[J, np.arange(n_obj)] +
            dist_to_next[J, np.arange(n_obj)],
            axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding


def randomized_argsort(A, method='numpy', order='ascending'):
    if method == 'numpy':
        P = np.random.permutation(len(A))
        Index = np.argsort(A[P], kind='quicksort')
        Index = P[Index]

    elif method == 'quicksort':
        Index = quicksort(A)

    else:
        raise Exception('Randomized sort method not known.')

    if order == 'ascending':
        return Index
    elif order == 'descending':
        return np.flip(Index, axis=0)
    else:
        raise Exception('Unknown sorting order: ascending or descending.')


def swap(M, a, b):
    tmp = M[a]
    M[a] = M[b]
    M[b] = tmp


def quicksort(A):
    Index = np.arange(len(A))
    _quicksort(A, Index, 0, len(A) - 1)
    return Index


def _quicksort(A, Index, left, right):
    if left < right:

        index = np.random.randint(left, right + 1)
        swap(Index, right, index)

        pivot = A[Index[right]]

        Index = left - 1

        for j in range(left, right):

            if A[Index[j]] <= pivot:
                Index += 1
                swap(Index, Index, j)

        index = Index + 1
        swap(Index, right, index)

        _quicksort(A, Index, left, index - 1)
        _quicksort(A, Index, index + 1, right)


def random_permuations(n, input):
    perms = []
    for _ in range(n):
        perms.append(np.random.permutation(input))
    P = np.concatenate(perms)
    return P


def crossover_mask(X, M):
    # convert input to output by flatting along the first axis
    _X = np.copy(X)
    _X[0][M] = X[1][M]
    _X[1][M] = X[0][M]

    return _X


def at_least_2d_array(x, extend_as='row'):
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 1:
        if extend_as == 'row':
            x = x[None, :]
        elif extend_as == 'column':
            x = x[:, None]

    return x


def repair_out_of_bounds(problem, X):
    xl, xu = problem.xl, problem.xu

    only_1d = (X.ndim == 1)
    X = at_least_2d_array(X)

    if xl is not None:
        xl = np.repeat(xl[None, :], X.shape[0], axis=0)
        X[X < xl] = xl[X < xl]

    if xu is not None:
        xu = np.repeat(xu[None, :], X.shape[0], axis=0)
        X[X > xu] = xu[X > xu]

    if only_1d:
        return X[0, :]
    else:
        return X


def denormalize(x, x_min, x_max):

    if x_max is None:
        _range = 1
    else:
        _range = (x_max - x_min)

    return x * _range + x_min


class Evaluator:
    """The evaluator class which is used during the algorithm execution to
    limit the number of evaluations."""

    def __init__(self, evaluate_values_of=['F', 'CV', 'G']):
        self.n_eval = 0
        self.evaluate_values_of = evaluate_values_of

    def eval(self, problem, pop, **kwargs):
        """This function is used to return the result of one valid evaluation.

        Parameters
        ----------
        problem : class
            The problem which is used to be evaluated
        pop : np.array or Population object
        kwargs : dict
            Additional arguments which might be necessary for the problem to
            evaluate.
        """

        is_individual = isinstance(pop, Individual)
        is_numpy_array = isinstance(
            pop, np.ndarray) and not isinstance(pop, Population)

        # make sure the object is a population
        if is_individual or is_numpy_array:
            pop = Population().create(pop)

        # find indices to be evaluated
        Index = [k for k in range(len(pop)) if pop[k].F is None]

        # update the function evaluation counter
        self.n_eval += len(Index)

        # actually evaluate all solutions using the function
        if len(Index) > 0:
            self._eval(problem, pop[Index], **kwargs)

            # set the feasibility attribute if cv exists
            for ind in pop[Index]:
                cv = ind.get('CV')
                if cv is not None:
                    ind.set('feasible', cv <= 0)

        if is_individual:
            return pop[0]
        elif is_numpy_array:
            if len(pop) == 1:
                pop = pop[0]
            return tuple([pop.get(e) for e in self.evaluate_values_of])
        else:
            return pop

    def _eval(self, problem, pop, **kwargs):

        out = problem.evaluate(
            pop.get('X'),
            return_values_of=self.evaluate_values_of,
            return_as_dictionary=True,
            **kwargs)

        for key, val in out.items():
            if val is None:
                continue
            else:
                pop.set(key, val)
