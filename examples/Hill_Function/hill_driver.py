from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

import themis
from trata import composite_samples, sampler
from ibis import mcmc, plots

MODEL_TYPE = 'gpr'
N = 150

GEN_DATA = False


def hill(x, a, b, c):
    return (a * x ** c) / (b ** c + x ** c)


def gen_hill_obs(x, a, b, c, obs_std, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.normal(hill(x, a, b, c), obs_std)


X_obs = [1.7995,
         0.83629,
         2.3971,
         2.2857,
         1.1870,
         1.9797,
         1.2773,
         1.3973,
         0.64994,
         1.1062,
         1.0851]

if GEN_DATA:
    Y_obs = gen_hill_obs(np.array(X_obs), 3.23, 0.66, 8.40, .08)
else:
    Y_obs = [3.4459,
             2.7616,
             3.0697,
             3.4208,
             2.9719,
             3.1330,
             3.5070,
             3.0320,
             1.4260,
             3.3516,
             3.4861]

###########################
# SETUP AND RUN ENSEMBLES #
###########################

mcmc_ana = mcmc.DefaultMCMC()

for i, (X_ob, Y_ob) in enumerate(zip(X_obs, Y_obs)):
    Samples = composite_samples.Samples()
    Samples.set_continuous_variable('a', 2, 2, 10)
    Samples.set_continuous_variable('b', 0, 0, 3)
    Samples.set_continuous_variable('c', 1, 1, 20)
    Samples.set_continuous_variable('x', 0, X_ob, 3)

    Samples.generate_samples(['a', 'b', 'c'], sampler.LatinHyperCubeSampler(), num_points=N)
    Samples.generate_samples(['x'], sampler.DefaultValueSampler(), num_points=N)

    print(Samples)
    runs = [themis.Run(sample, args='--input_deck input_deck') for sample in Samples]

    mgr = themis.Themis.create_overwrite(
        os.path.join(os.path.dirname(__file__), 'hill_function.exe'),
        runs=runs,
        run_dir_names=os.path.join('ens_' + str(i), '{run_id}'),
        app_interface=os.path.join(os.path.dirname(__file__), 'hill_interface.py'),
        run_parse=os.path.join(os.path.dirname(__file__), 'input_deck'),
        app_is_batch_script=False,
        setup_dir=".ens_" + str(i) + "_setup",
    )
    mgr.execute_local(blocking=True)
    print(mgr.progress())

    ###########################
    # CREATE SURROGATE MODELS #
    ###########################

    Y = mgr.as_dataframe(include_none=True)[["result"]].to_numpy()

    points = Samples.get_points(['a', 'b', 'c'], scaled=False)

    model = GaussianProcessRegressor()
    model.fit(points, Y)

    print(model)

    mcmc_ana.add_output('{}_{}', 'Y', model, Y_ob, .08, ['a', 'b', 'c'])

################
# RUN ANALYSIS #
################

mcmc_ana.add_input('a', 2, 10, .04)
mcmc_ana.add_input('b', 0, 3, .04)
mcmc_ana.add_input('c', 1, 20, .2)

mcmc_ana.run_chain(total=10000, burn=10000, every=2, n_chains=16, prior_only=True, seed=20200824)
print(mcmc_ana.diagnostics_string())

prior_chains = mcmc_ana.get_chains(flattened=True)
prior_points = np.stack(prior_chains.values()).T

for name in ['a', 'b', 'c']:
    fig, ax = plt.subplots(1, 1)
    mcmc_ana.trace_plot(name, ax=ax)

    fig.savefig('{}_prior_trace_plot.png'.format(name))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    mcmc_ana.autocorr_plot(name, ax=ax)

    fig.savefig('{}_prior_autocorr_plot.png'.format(name))
    plt.close(fig)

mcmc_ana.run_chain(total=10000, burn=10000, every=30, n_chains=16, prior_only=False, seed=20200825)
print(mcmc_ana.diagnostics_string())

post_chains = mcmc_ana.get_chains(flattened=True)
post_points = np.stack(post_chains.values()).T

for name in ['a', 'b', 'c']:
    fig, ax = plt.subplots(1, 1)
    mcmc_ana.trace_plot(name, ax=ax)

    fig.savefig('{}_post_trace_plot.png'.format(name))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    mcmc_ana.autocorr_plot(name, ax=ax)

    fig.savefig('{}_post_autocorr_plot.png'.format(name))
    plt.close(fig)

for key in post_chains.keys():
    fig, ax = plt.subplots(1, 1)
    plots.likelihood_plot(ax, prior_chains[key], post_points=post_chains[key])

    fig.savefig('{}_likelihood.png'.format(key))
    plt.close(fig)
