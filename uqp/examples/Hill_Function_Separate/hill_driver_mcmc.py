from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('/collab/usr/gapps/uq/')

from uqp.surrogate_model import surrogate_model
from uqp.uq_component import uqp_mcmc

import matplotlib.pyplot as plt

import hill_data

mcmc_ana = uqp_mcmc.DefaultMCMC()

mcmc_ana.add_input('a', 2, 10, 2.0)
mcmc_ana.add_input('b', 0, 3, 0.5)
mcmc_ana.add_input('c', 1, 20, 5.5)

for i, (X_ob, Y_ob) in enumerate(zip(hill_data.X_obs, hill_data.Y_obs)):
    model = surrogate_model.SurrogateModel.load('{}_{}.mdl'.format(i, hill_data.MODEL_TYPE))

    mcmc_ana.add_output('{}_{}', 'Y', model, Y_ob, .08, ['a', 'b', 'c'])

mcmc_ana.run_chain(total=2500, burn=100, every=30, n_chains=16, prior_only=True, seed=20200824)
print(mcmc_ana.diagnostics_string())

for name in ['a', 'b', 'c']:
    fig, ax = plt.subplots(1, 1)
    mcmc_ana.trace_plot(name, ax=ax)

    fig.savefig('{}_prior_trace_plot.png'.format(name))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    mcmc_ana.autocorr_plot(name, ax=ax)

    fig.savefig('{}_prior_autocorr_plot.png'.format(name))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    mcmc_ana.histogram_plot(name, ax=ax, bins=200)

    fig.savefig('{}_prior_hist_plot.png'.format(name))
    plt.close(fig)

mcmc_ana.run_chain(total=2500, burn=100, every=30, n_chains=16, prior_only=False, seed=20200825)
print(mcmc_ana.diagnostics_string())

for name in ['a', 'b', 'c']:
    fig, ax = plt.subplots(1, 1)
    mcmc_ana.trace_plot(name, ax=ax)

    fig.savefig('{}_post_trace_plot.png'.format(name))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    mcmc_ana.autocorr_plot(name, ax=ax)

    fig.savefig('{}_post_autocorr_plot.png'.format(name))
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    mcmc_ana.histogram_plot(name, ax=ax, bins=200)

    fig.savefig('{}_post_hist_plot.png'.format(name))
    plt.close(fig)

# fig, ax = plt.subplots(1, 1)
# mcmc_ana.posterior_predictive_plot('Y_0', ax=ax)
# fig.savefig('post_prediction.png')
# plt.close(fig)
