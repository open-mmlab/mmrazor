# Copyright (c) OpenMMLab. All rights reserved.
# copied and modified from https://github.com/anyoptimization/pymoo
from abc import abstractmethod

import numpy as np
from pymoo.util.optimum import filter_optimum

# from pymoo.core.evaluator import Evaluator
# from pymoo.core.population import Population
from .utils.helper import Evaluator, Population


class BaseOptimizer():
    """This class represents the abstract class for any algorithm to be
    implemented. The solve method provides a wrapper function which does
    validate the input.

    Args:
        problem :
            Problem to be solved by the algorithm
        verbose (bool):
            If true information during the algorithm execution are displayed
        save_history (bool):
            If true, a current snapshot of each generation is saved.
        pf (numpy.array):
            The Pareto-front for the given problem. If provided performance
            metrics are printed during execution.
        return_least_infeasible (bool):
            Whether the algorithm should return the least infeasible solution,
            if no solution was found.
        evaluator : :class:`~pymoo.model.evaluator.Evaluator`
            The evaluator which can be used to make modifications before
            calling the evaluate function of a problem.
    """

    def __init__(self, **kwargs):
        # !
        # DEFAULT SETTINGS OF ALGORITHM
        # !
        # set the display variable supplied to the algorithm
        self.display = kwargs.get('display')
        self.logger = kwargs.get('logger')
        # !
        # Attributes to be set later on for each problem run
        # !
        # the optimization problem as an instance
        self.problem = None

        self.return_least_infeasible = None
        # whether the history should be saved or not
        self.save_history = None
        # whether the algorithm should print output in this run or not
        self.verbose = None
        # the random seed that was used
        self.seed = None
        # the pareto-front of the problem - if it exist or passed
        self.pf = None
        # the function evaluator object (can be used to inject code)
        self.evaluator = None
        # the current number of generation or iteration
        self.n_gen = None
        # the history object which contains the list
        self.history = None
        # the current solutions stored - here considered as population
        self.pop = None
        # the optimum found by the algorithm
        self.opt = None
        # can be used to store additional data in submodules
        self.data = {}

    def initialize(
            self,
            problem,
            pf=True,
            evaluator=None,
            # START Default minimize
            seed=None,
            verbose=False,
            save_history=False,
            return_least_infeasible=False,
            # END Default minimize
            n_gen=1,
            display=None,
            # END Overwrite by minimize
            **kwargs):

        # set the problem that is optimized for the current run
        self.problem = problem

        # set the provided pareto front
        self.pf = pf

        # by default make sure an evaluator exists if nothing is passed
        if evaluator is None:
            evaluator = Evaluator()
        self.evaluator = evaluator

        # !
        # START Default minimize
        # !
        # if this run should be verbose or not
        self.verbose = verbose
        # whether the least infeasible should be returned or not
        self.return_least_infeasible = return_least_infeasible
        # whether the history should be stored or not
        self.save_history = save_history

        # set the random seed in the algorithm object
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 10000000)
        np.random.seed(self.seed)
        # !
        # END Default minimize
        # !

        if display is not None:
            self.display = display

        # other run dependent variables that are reset
        self.n_gen = n_gen
        self.history = []
        self.pop = Population()
        self.opt = None

    def solve(self):

        # the result object to be finally returned
        res = {}

        # initialize the first population and evaluate it
        self._initialize()
        self._set_optimum()

        self.current_gen = 0
        # while termination criterion not fulfilled
        while self.current_gen < self.n_gen:
            self.current_gen += 1
            self.next()

        # store the resulting population
        res['pop'] = self.pop

        # get the optimal solution found
        opt = self.opt

        # if optimum is not set
        if len(opt) == 0:
            opt = None

        # if no feasible solution has been found
        elif not np.any(opt.get('feasible')):
            if self.return_least_infeasible:
                opt = filter_optimum(opt, least_infeasible=True)
            else:
                opt = None

        # set the optimum to the result object
        res['opt'] = opt

        # if optimum is set to none to not report anything
        if opt is None:
            X, F, CV, G = None, None, None, None

        # otherwise get the values from the population
        else:
            X, F, CV, G = self.opt.get('X', 'F', 'CV', 'G')

            # if single-objective problem and only one solution was found
            if self.problem.n_obj == 1 and len(X) == 1:
                X, F, CV, G = X[0], F[0], CV[0], G[0]

        # set all the individual values
        res['X'], res['F'], res['CV'], res['G'] = X, F, CV, G

        # create the result object
        res['problem'], res['pf'] = self.problem, self.pf
        res['history'] = self.history

        return res

    def next(self):
        # call next of the implementation of the algorithm
        self._next()

        # set the optimum - only done if the algorithm did not do it yet
        self._set_optimum()

        # do what needs to be done each generation
        self._each_iteration()

    # method that is called each iteration to call some algorithms regularly
    def _each_iteration(self):

        # display the output if defined by the algorithm
        if self.logger:
            self.logger.info(f'Generation:[{self.current_gen}/{self.n_gen}] '
                             f'evaluate {self.evaluator.n_eval} solutions, '
                             f'find {len(self.opt)} optimal solution.')

    def _finalize(self):
        pass

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _next(self):
        pass
