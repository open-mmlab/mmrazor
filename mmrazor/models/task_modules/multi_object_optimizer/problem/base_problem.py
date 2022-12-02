# Copyright (c) OpenMMLab. All rights reserved.
# copied and modified from https://github.com/anyoptimization/pymoo
from abc import abstractmethod

import numpy as np


def at_least_2d_array(x, extend_as='row'):
    if not isinstance(x, np.ndarray):
        x = np.array([x])

    if x.ndim == 1:
        if extend_as == 'row':
            x = x[None, :]
        elif extend_as == 'column':
            x = x[:, None]

    return x


class BaseProblem():
    """Superclass for each problem that is defined.

    It provides attributes such as the number of variables, number of
    objectives or constraints. Also, the lower and upper bounds are stored. If
    available the Pareto-front, nadir point and ideal point are stored.
    """

    def __init__(self,
                 n_var=-1,
                 n_obj=-1,
                 n_constr=0,
                 xl=None,
                 xu=None,
                 type_var=np.double,
                 evaluation_of='auto',
                 parallelization=None,
                 elementwise_evaluation=False,
                 callback=None):
        """
        Args:
            n_var (int):
                number of variables
            n_obj (int):
                number of objectives
            n_constr (int):
                number of constraints
            xl (np.array or int):
                lower bounds for the variables.
            xu (np.array or int):
                upper bounds for the variables.
            type_var (numpy type):
                type of the variable to be evaluated.
            elementwise_evaluation (bool):
            parallelization (str or tuple):
                See :ref:`nb_parallelization` for guidance on parallelization.

        """

        # number of variable for this problem
        self.n_var = n_var

        # type of the variable to be evaluated
        self.type_var = type_var

        # number of objectives
        self.n_obj = n_obj

        # number of constraints
        self.n_constr = n_constr

        # allow just an integer for xl and xu if all bounds are equal
        if n_var > 0 and not isinstance(xl, np.ndarray) and xl is not None:
            self.xl = np.ones(n_var) * xl
        else:
            self.xl = xl

        if n_var > 0 and not isinstance(xu, np.ndarray) and xu is not None:
            self.xu = np.ones(n_var) * xu
        else:
            self.xu = xu

        # the pareto set and front will be calculated only once.
        self._pareto_front = None
        self._pareto_set = None
        self._ideal_point, self._nadir_point = None, None

        # actually defines what _evaluate is setting during the evaluation
        if evaluation_of == 'auto':
            # by default F is set, and G if the problem does have constraints
            self.evaluation_of = ['F']
            if self.n_constr > 0:
                self.evaluation_of.append('G')
        else:
            self.evaluation_of = evaluation_of

        self.elementwise_evaluation = elementwise_evaluation

        self.parallelization = parallelization

        # store the callback if defined
        self.callback = callback

    def nadir_point(self):
        """Return nadir_point (np.array):

        The nadir point for a multi-objective problem.
        """
        # if the ideal point has not been calculated yet
        if self._nadir_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self.pareto_front()

            # if already done or it was successful - calculate the ideal point
            if self._pareto_front is not None:
                self._ideal_point = np.max(self._pareto_front, axis=0)

        return self._nadir_point

    def ideal_point(self):
        """
        Returns
        -------
        ideal_point (np.array):
            The ideal point for a multi-objective problem. If single-objective
            it returns the best possible solution.
        """

        # if the ideal point has not been calculated yet
        if self._ideal_point is None:

            # calculate the pareto front if not happened yet
            if self._pareto_front is None:
                self.pareto_front()

            # if already done or it was successful - calculate the ideal point
            if self._pareto_front is not None:
                self._ideal_point = np.min(self._pareto_front, axis=0)

        return self._ideal_point

    def pareto_front(self,
                     *args,
                     use_cache=True,
                     exception_if_failing=True,
                     **kwargs):
        """
        Args:
            args : Same problem implementation need some more information to
                    create the Pareto front.
            exception_if_failing (bool):
                Whether to throw an exception when generating the Pareto front
                has failed.
            use_cache (bool):
                    Whether to use the cache if the Pareto front.

        Returns:
            P (np.array):
                The Pareto front of a given problem.

        """
        if not use_cache or self._pareto_front is None:
            try:
                pf = self._calc_pareto_front(*args, **kwargs)
                if pf is not None:
                    self._pareto_front = at_least_2d_array(pf)

            except Exception as e:
                if exception_if_failing:
                    raise e

        return self._pareto_front

    def pareto_set(self, *args, use_cache=True, **kwargs):
        """
        Returns:
            S (np.array):
                Returns the pareto set for a problem.
        """
        if not use_cache or self._pareto_set is None:
            self._pareto_set = at_least_2d_array(
                self._calc_pareto_set(*args, **kwargs))

        return self._pareto_set

    def evaluate(self,
                 X,
                 *args,
                 return_values_of='auto',
                 return_as_dictionary=False,
                 **kwargs):
        """Evaluate the given problem.

        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible.

        Args:

            X (np.array):
                A two dimensional matrix where each row is a point to evaluate
                and each column a variable.

            return_as_dictionary (bool):
                If this is true than only one object, a dictionary,
                is returned.
            return_values_of (list of strings):
                Allowed is ["F", "CV", "G", "dF", "dG", "dCV", "feasible"]
                where the d stands for derivative and h stands for hessian
                matrix.


        Returns:
            A dictionary, if return_as_dictionary enabled, or a list of values
            as defined in return_values_of.
        """

        # call the callback of the problem
        if self.callback is not None:
            self.callback(X)

        only_single_value = len(np.shape(X)) == 1
        X = np.atleast_2d(X)

        # check the dimensionality of the problem and the given input
        if X.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' %
                            (X.shape[1], self.n_var))

        if type(return_values_of) == str and return_values_of == 'auto':
            return_values_of = ['F']
            if self.n_constr > 0:
                return_values_of.append('CV')

        out = {}
        for val in return_values_of:
            out[val] = None

        out = self._evaluate_batch(X, False, out, *args, **kwargs)

        # if constraint violation should be returned as well
        if self.n_constr == 0:
            CV = np.zeros([X.shape[0], 1])
        else:
            CV = self.calc_constraint_violation(out['G'])

        if 'CV' in return_values_of:
            out['CV'] = CV

        # if an additional boolean flag for feasibility should be returned
        if 'feasible' in return_values_of:
            out['feasible'] = (CV <= 0)

        # if asked for a value but not set in the evaluation set to None
        for val in return_values_of:
            if val not in out:
                out[val] = None

        if only_single_value:
            for key in out.keys():
                if out[key] is not None:
                    out[key] = out[key][0, :]

        if return_as_dictionary:
            return out
        else:

            # if just a single value do not return a tuple
            if len(return_values_of) == 1:
                return out[return_values_of[0]]
            else:
                return tuple([out[val] for val in return_values_of])

    def _evaluate_batch(self, X, calc_gradient, out, *args, **kwargs):
        self._evaluate(X, out, *args, **kwargs)
        for key in out.keys():
            if len(np.shape(out[key])) == 1:
                out[key] = out[key][:, None]

        return out

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        pass

    def has_bounds(self):
        return self.xl is not None and self.xu is not None

    def bounds(self):
        return self.xl, self.xu

    def name(self):
        return self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        """Method that either loads or calculates the pareto front. This is
        only done ones and the pareto front is stored.

        Returns:
            pf (np.array): Pareto front as array.
        """
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass

    # some problem information
    def __str__(self):
        s = '# name: %s\n' % self.name()
        s += '# n_var: %s\n' % self.n_var
        s += '# n_obj: %s\n' % self.n_obj
        s += '# n_constr: %s\n' % self.n_constr
        s += '# f(xl): %s\n' % self.evaluate(self.xl)[0]
        s += '# f((xl+xu)/2): %s\n' % self.evaluate(
            (self.xl + self.xu) / 2.0)[0]
        s += '# f(xu): %s\n' % self.evaluate(self.xu)[0]
        return s

    @staticmethod
    def calc_constraint_violation(G):
        if G is None:
            return None
        elif G.shape[1] == 0:
            return np.zeros(G.shape[0])[:, None]
        else:
            return np.sum(G * (G > 0), axis=1)[:, None]
