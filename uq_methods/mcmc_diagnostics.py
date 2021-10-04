from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from itertools import count

import numpy as np


def get_chains_size(chains):
    chains = np.array(chains)
    chains_shape = chains.shape
    if len(chains_shape) != 2:
        raise Exception("Given chains are not all the same length")

    return chains_shape


def within_variance(chains):
    # mean of variances
    return np.mean([np.var(chain, ddof=1) for chain in chains])


def between_variance(chains):
    # variance of means
    m, n = get_chains_size(chains)

    return np.var([np.mean(chain) for chain in chains], ddof=1) * n


def var_hat(chains):
    n = len(chains[0])

    return ((n - 1.) / n) * within_variance(chains) + (between_variance(chains) / n)


def r_hat(chains):
    return np.sqrt(var_hat(chains) / within_variance(chains))


def variogram(chains, t):
    total_mean = np.mean([np.mean(np.power(chain[:-t] - chain[t:], 2)) for chain in chains])
    return total_mean


def rho_hat(chains, t, prev_var_hat=None):
    if prev_var_hat is None:
        return 1. - variogram(chains, t) / (2 * var_hat(chains))
    else:
        return 1. - variogram(chains, t) / (2 * prev_var_hat)


def n_eff(chains):
    m, n = get_chains_size(chains)
    prev_var_hat = var_hat(chains)

    s = 0.0

    it1 = (rho_hat(chains, t, prev_var_hat) for t in count(1))
    for x in it1:
        val = x + next(it1)
        if val >= 0:
            s += val
        else:
            break

    return m * n / (1 + 2 * s)


def split(chain, n_splits=2):
    def chunk():
        n = len(chain) // n_splits
        for i in range(0, len(chain), n):
            yield chain[i:i + n]

    return np.array(list(chunk()))


def mode(chains, bins=100):
    height, bins = np.histogram(chains, bins=bins)
    max_bin = np.argmax(height)
    return np.mean([bins[max_bin], bins[max_bin+1]])


