import itertools

import numpy as np
import scipy as sp
import pandas as pd

from src.oom import ContinuousValuedOOM
from src.oom.discrete_observable import DiscreteObservable


def fix_pvec(p, epsilon=1e-20):
    p[p <= 0] = epsilon
    p /= np.sum(p)
    return p

def kl_divergence(p, q):
    return sp.stats.entropy(p, q)

def mse(p, q):
    return np.sum((p - q) ** 2)


def quantify_distribution(
    steps: int,
    state: np.matrix,
    operators: list[np.matrix],
    lin_func: np.matrix
) -> np.array:
    """
    
    """
    obsops = dict(zip(range(len(operators)), operators))
    pred_dists = {}
    
    for obscomb in itertools.product(obsops.keys(), repeat = steps):
        curr_state = state
        
        for obs in reversed(obscomb):
            curr_state = obsops[obs] * curr_state
        
        curr_p = lin_func * curr_state
        pred_dists[obscomb] = curr_p.item()
    
    # Order and return results
    pred_dists_ps = pd.Series(pred_dists)
    # print('\n', pred_dists_ps, '\n\n', pred_dists_ps.sum(), sep='')
    return pred_dists_ps.values


def get_discretized_operators(
    oom: ContinuousValuedOOM,
    intervals: list[tuple[float, float]],
) -> dict[DiscreteObservable, np.matrix]:
    """
    
    """
    obsops = {}
    for idx, interval in enumerate(intervals):
        il, ir = interval
        
        # Get operator over this interval
        wops = []
        for mf, op in zip(oom.membership_fns, oom.operators):
            weight = mf.cdf(ir) - mf.cdf(il)
            wop = weight * op
            wops.append(wop)
        op = sum(wops)
        
        obsops[DiscreteObservable(name=f"L{idx}_{interval}")] = op
    return obsops