# Copyright (c) OpenMMLab. All rights reserved.
# copied and modified from https://github.com/anyoptimization/pymoo
import numpy as np

from mmrazor.registry import TASK_UTILS
from .base_optimizer import BaseOptimizer
from .utils.selection import (IntegerFromFloatMutation, Mating,
                              PointCrossover, TournamentSelection,
                              binary_tournament)
from .utils.helper import (DefaultDuplicateElimination, Individual,
                           Initialization, Survival)

# from pymoo.algorithms.moo.nsga2 import binary_tournament
# from pymoo.core.mating import Mating
# from pymoo.core.survival import Survival
# from pymoo.core.individual import Individual
# from pymoo.core.initialization import Initialization
# from pymoo.core.duplicate import DefaultDuplicateElimination
# from pymoo.operators.crossover.pntx import PointCrossover
# from pymoo.operators.selection.tournament import TournamentSelection
# from .packages.selection import IntegerFromFloatMutation


@TASK_UTILS.register_module()
class NSGA2Optimizer(BaseOptimizer):
    """Implementation of NSGA2 search method.

    Args:
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}
    """

    def __init__(self,
                 pop_size=100,
                 sampling=None,
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=PointCrossover(n_points=2),
                 mutation=IntegerFromFloatMutation(eta=1.0),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=None,
                 survival=Survival(),
                 repair=None,
                 **kwargs):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            display=display,
            **kwargs)

        # the population size used
        self.pop_size = pop_size

        # the survival for the genetic algorithm
        self.survival = Survival()

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # the object to be used to represent an individual
        self.individual = Individual(rank=np.inf, crowding=-1)

        # set the duplicate detection class
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = None
        else:
            self.eliminate_duplicates = eliminate_duplicates

        self.initialization = Initialization(
            sampling,
            individual=self.individual,
            repair=repair,
            eliminate_duplicates=self.eliminate_duplicates)

        self.mating = Mating(
            selection,
            crossover,
            mutation,
            repair=repair,
            eliminate_duplicates=self.eliminate_duplicates,
            n_max_iterations=100)

        # other run specific data updated whenever solve is called
        self.n_gen = None
        self.pop = None
        self.off = None

    def _initialize(self):

        # create the initial population
        pop = self.initialization.do(
            self.problem, self.pop_size, algorithm=self)

        # then evaluate using the objective function
        self.evaluator.eval(self.problem, pop, algorithm=self)

        # that call is a dummy survival to set attributes
        # that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self)

        self.pop, self.off = pop, pop

    def _next(self):

        # do the mating using the current population
        self.off = self.mating.do(
            self.problem,
            self.pop,
            n_offsprings=self.n_offsprings,
            algorithm=self)

        # if the mating could not generate any new offspring
        if len(self.off) == 0:
            return

        # evaluate the offspring
        self.evaluator.eval(self.problem, self.off, algorithm=self)

        # merge the offsprings with the current population
        self.pop = self.pop.merge(self.off)

        # the do survival selection
        if self.survival:
            self.pop = self.survival.do(
                self.problem, self.pop, self.pop_size, algorithm=self)

    def _set_optimum(self, **kwargs):
        if not np.any(self.pop.get('feasible')):
            self.opt = self.pop[[np.argmin(self.pop.get('CV'))]]
        else:
            self.opt = self.pop[self.pop.get('rank') == 0]
