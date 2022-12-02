# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from pymoo.config import Config
from pymoo.core.decision_making import (DecisionMaking, NeighborFinder,
                                        find_outliers_upper_tail)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

Config.warnings['not_compiled'] = False


class HighTradeoffPoints(DecisionMaking):
    """Method for multi-object optimization.

    Args:
        ratio(float): weight between score_key and sec_obj, details in
        demo/nas/demo.ipynb.
        epsilon(float): specific a radius for each neighbour.
        n_survive(int): how many high-tradeoff points will return finally.
    """

    def __init__(self,
                 ratio=1,
                 epsilon=0.125,
                 n_survive=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive
        self.ratio = ratio

    def _do(self, data, **kwargs):
        front = NonDominatedSorting().do(data, only_non_dominated_front=True)
        F = data[front, :]

        n, m = F.shape
        F = normalize(F, self.ideal, self.nadir)
        F[:, 1] = F[:, 1] * self.ratio

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

        # if given topk
        if self.n_survive is not None:
            n_survive = min(self.n_survive, len(mu))
            index = np.argsort(mu)[-n_survive:][::-1]
            front_survive = front[index]

            self.n_survive -= n_survive
            if self.n_survive == 0:
                return front_survive
            # in case the survived in front is not enough for topk
            index = np.array(list(set(np.arange(len(data))) - set(front)))
            unused_data = data[index]
            no_front_survive = index[self._do(unused_data)]

            return np.concatenate([front_survive, no_front_survive])
        else:
            # return points with trade-off > 2*sigma
            mu = find_outliers_upper_tail(mu)
            return mu if len(mu) else []
