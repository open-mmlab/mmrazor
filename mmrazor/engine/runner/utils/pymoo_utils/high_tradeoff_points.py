# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

try:
    from pymoo.configuration import Configuration
    from pymoo.model.decision_making import (DecisionMaking, NeighborFinder,
                                             find_outliers_upper_tail,
                                             normalize)
    Configuration.show_compile_hint = False
except ImportError:
    from mmrazor.utils import get_placeholder
    DecisionMaking = get_placeholder('pymoo')
    NeighborFinder = get_placeholder('pymoo')
    normalize = get_placeholder('pymoo')
    find_outliers_upper_tail = get_placeholder('pymoo')
    Configuration = get_placeholder('pymoo')


class HighTradeoffPoints(DecisionMaking):
    """Method for multi-object optimization.

    Args:
        ratio(float): weight between score_key and sec_obj, details in
        demo/nas/demo.ipynb.
        epsilon(float): specific a radius for each neighbour.
        n_survive(int): how many high-tradeoff points will return finally.
    """

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(
                F,
                self.ideal_point,
                self.nadir_point,
                estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(
            F, epsilon=0.125, n_min_neigbors='auto', consider_2d=False)

        mu = np.full(n, -np.inf)

        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)

        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            # return points with trade-off > 2*sigma
            return find_outliers_upper_tail(mu)
