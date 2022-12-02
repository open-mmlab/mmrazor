# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def get_relation(a, b, cva=None, cvb=None):

    if cva is not None and cvb is not None:
        if cva < cvb:
            return 1
        elif cvb < cva:
            return -1

    val = 0
    for i in range(len(a)):
        if a[i] < b[i]:
            # indifferent because once better and once worse
            if val == -1:
                return 0
            val = 1
        elif b[i] < a[i]:
            # indifferent because once better and once worse
            if val == 1:
                return 0
            val = -1
    return val


def calc_domination_matrix_loop(F, G):
    n = F.shape[0]
    CV = np.sum(G * (G > 0).astype(np.float32), axis=1)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = get_relation(F[i, :], F[j, :], CV[i], CV[j])
            M[j, i] = -M[i, j]

    return M


def calc_domination_matrix(F, _F=None, epsilon=0.0):
    """
    if G is None or len(G) == 0:
        constr = np.zeros((F.shape[0], F.shape[0]))
    else:
        # consider the constraint violation
        # CV = Problem.calc_constraint_violation(G)
        # constr = (CV < CV) * 1 + (CV > CV) * -1

        CV = Problem.calc_constraint_violation(G)[:, 0]
        constr = (CV[:, None] < CV) * 1 + (CV[:, None] > CV) * -1
    """

    if _F is None:
        _F = F

    # look at the obj for dom
    n = F.shape[0]
    m = _F.shape[0]

    L = np.repeat(F, m, axis=0)
    R = np.tile(_F, (n, 1))

    smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
    larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

    M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
        + np.logical_and(larger, np.logical_not(smaller)) * -1

    # if cv equal then look at dom
    # M = constr + (constr == 0) * dom

    return M


def fast_non_dominated_sort(F, **kwargs):
    M = calc_domination_matrix(F)

    # calculate the dominance matrix
    n = M.shape[0]

    fronts = []

    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.zeros(n, dtype=np.int32)
    is_dominating = [[] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = np.zeros(n)

    current_front = []

    for i in range(n):

        for j in range(i + 1, n):
            rel = M[i, j]
            if rel == 1:
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif rel == -1:
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:

        next_front = []

        # for each individual in the current front
        for i in current_front:

            # all solutions that are dominated by this individuals
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts
