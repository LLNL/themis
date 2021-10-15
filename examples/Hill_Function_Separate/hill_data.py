from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel

MODEL_TYPE = GaussianProcessRegressor
MODEL_ARGS = {'kernel': WhiteKernel() + ConstantKernel() * RBF()}
N = 1000

GEN_DATA = False


def hill(x, a, b, c):
    return (a * x ** c) / (b ** c + x ** c)


def gen_hill_obs(x, a, b, c, obs_std, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.normal(hill(x, a, b, c), obs_std)


X_obs = np.array([1.7995,
                  0.83629,
                  2.3971,
                  2.2857,
                  1.1870,
                  1.9797,
                  1.2773,
                  1.3973,
                  0.64994,
                  1.1062,
                  1.0851])

if GEN_DATA:
    Y_obs = gen_hill_obs(X_obs, 3.23, 0.66, 8.40, .08)
else:
    Y_obs = np.array([3.4459,
                      2.7616,
                      3.0697,
                      3.4208,
                      2.9719,
                      3.1330,
                      3.5070,
                      3.0320,
                      1.4260,
                      3.3516,
                      3.4861])
