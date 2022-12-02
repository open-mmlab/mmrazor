# Copyright (c) OpenMMLab. All rights reserved.
# copied and modified from https://github.com/anyoptimization/pymoo
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mmrazor.registry import TASK_UTILS
from .nsga2_optimizer import NSGA2Optimizer
from .utils.helper import Individual, Population
from .utils.selection import (BinaryCrossover, IntegerFromFloatMutation, Mating,
                              MyMutation, MySampling, PointCrossover,
                              TournamentSelection)



@TASK_UTILS.register_module()
class GeneticOptimizer(NSGA2Optimizer):
    """Genetic Algorithm."""

    def __init__(self,
                 pop_size=100,
                 sampling=MySampling(),
                 selection=TournamentSelection(func_comp='comp_by_cv_and_fitness'),
                 crossover=BinaryCrossover(),
                 mutation=MyMutation(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=None,
                 **kwargs):
        """
        Args:
            pop_size : {pop_size}
            sampling : {sampling}
            selection : {selection}
            crossover : {crossover}
            mutation : {mutation}
            eliminate_duplicates : {eliminate_duplicates}
            n_offsprings : {n_offsprings}

        """

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=FitnessSurvival(),
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            display=display,
            **kwargs)

    def _set_optimum(self, force=False):
        pop = self.pop
        self.opt = filter_optimum(pop, least_infeasible=True)


def filter_optimum(pop, least_infeasible=False):
    # first only choose feasible solutions
    ret = pop[pop.get('feasible')[:, 0]]

    # if at least one feasible solution was found
    if len(ret) > 0:

        # then check the objective values
        F = ret.get('F')

        if F.shape[1] > 1:
            Index = NonDominatedSorting().do(F, only_non_dominated_front=True)
            ret = ret[Index]

        else:
            ret = ret[np.argmin(F)]

    # no feasible solution was found
    else:
        # if flag enable report the least infeasible
        if least_infeasible:
            ret = pop[np.argmin(pop.get('CV'))]
        # otherwise just return none
        else:
            ret = None

    if isinstance(ret, Individual):
        ret = Population().create(ret)

    return ret
