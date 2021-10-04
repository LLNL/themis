from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TKagg')

import functools
import logging
import multiprocessing
import sys
import time
import traceback
from copy import deepcopy

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import scipy.stats as sts

from uq_methods import mcmc_diagnostics


def time_it(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        s = "    {}: {:.6f} s".format(func.__name__, (end - start))
        print(s)
        logging.info(s)
        return result

    return timed


class MCMC(object):
    """
        An interface for setting up and running Markov Chain Monte Carlo.

        An interface for setting up and running Markov Chain Monte Carlo on a set of unobserved inputs and observed
        outputs with surrogate models mapping inputs to outputs.
        Sub classes must implement _run_chain()
    """

    def __init__(self):
        self.outputs = {}
        self.inputs = {}
        self.chains = None
        self.acceptance_rate = None
        self.log_prob = None

        logging.info('Instantiated {}'.format(type(self).__name__))

    def add_output(self, event, quantity, surrogate_model, observed_value, observed_std, inputs):
        """
            Add an output variable to the set of variables

            An output variable represents an observed output that depends on some unobserved inputs. The function that
            maps inputs to outputs is approximated with a surrogate model.

            :parameter name: The unique name of the output
            :type name: str

            :parameter event: The event the output is associated with
            :type event: str

            :parameter quantity: The physical quantity of the output
            :type quantity: str

            :parameter surrogate_model: The surrogate model that represents the mapping of input values to output
                                        values. This model must be have a predict method that takes a numpy array and
                                        returns a numpy array (sklearn's fit/predict paradigm)

            :parameter observed_value: The observed experimental value
            :type observed_value: float

            :parameter observed_std: The error bound on the observed value
            :type observed_std: float

            :parameter inputs: The ordered names of the input variables that map to the output
            :type inputs: list of str
        """
        self.outputs[event + '_' + quantity] = OutputVariable(event, quantity, surrogate_model, observed_value,
                                                              observed_std, inputs)

        logging.info('\nAdded Output\nEvent: {}\nQuantity: {}\nObs Mean: {}\nObs Std: {}\n'
                     'Inputs: {}\nSurrogate Model: {}'.format(event, quantity,
                                                              observed_value,
                                                              observed_std, inputs,
                                                              surrogate_model))

    def remove_output(self, name):
        """
            Remove an output variable from the set of variables

            :parameter name: The name of the output variable to remove
            :type name: str
        """

        del self.outputs[name]

    def add_input(self, name, low, high, proposal_sigma, prior=None, unscaled_low=None, unscaled_high=None, scaling=None):
        """
            Add an input variable to the set of variables

            An input variable represents an unobserved input for some unobserved outputs. An input can be used for 1 or
            many outputs

            :parameter name: The unique name of the input
            :type name: str

            :parameter low: The lower bound on the value this variable can take
            :type low: float

            :parameter high: The upper bound on the value this variable can take
            :type high: float

            :parameter proposal_sigma: The standard deviation of the proposal distribution for this variable
            :type proposal_sigma: float

            :parameter prior: The prior distribution of the variable.
                              If None, a uniform distribution over the whole range is used
            :type prior: function
        """
        self.inputs[name] = InputVariable(name, low, high, proposal_sigma, prior, unscaled_low, unscaled_high, scaling)
        logging.info('\nAdded Input\nName: {}\nLower Bound: {}\nUpper Bound: {}\n'
                     'Proposal Sigma: {}\nPrior Distribution: {}\n'
                     'Unscaled Lower Bound: {}\nUnscaled Upper Bound: {}\n'
                     'Scaling: {}'.format(name, low, high, proposal_sigma, prior, unscaled_low, unscaled_high, scaling))

    def remove_input(self, name):
        """
            Remove an input variable from the set of variables

            :parameter name: The name of the input variable to remove
            :type name: str
        """

        del self.inputs[name]

    @time_it
    def run_chain(self, total, burn, every, start=None, n_chains=-1, prior_only=False, seed=None):
        """
            Runs the MCMC Algorithm

            Number of Iterations = n_chains*(total*every + burn)

            :parameter total: The total number of sample points to return
            :type total: int

            :parameter burn: The number of burn-in iterations
            :type burn: int

            :parameter every: The rate at which to save points. Saves every Nth iteration
            :type every: int

            :parameter start: The value at which to start the chains
            :type start: dict of str, float

            :parameter n_chains: The number of chains to run in parallel
            :type n_chains: int

            :parameter prior_only: Whether to run the chain on just the prior distributions.
            :type prior_only: bool

            :parameter seed: The random seed for the chains
            :type seed: int
        """
        if seed is not None:
            np.random.seed(seed)

        if n_chains == -1:
            logging.info('\nRunning Chain\nTotal: {}\nBurn: {}\nEvery: {}'
                         '\nStart: {}\nPrior Only: {}\nSeed: {}'.format(total, burn, every,
                                                                        start, prior_only, seed))
            try:
                data = self._run_chain(np.random.randint(2 ** 32 - 1), total, burn, every, start, prior_only)
            except Exception as ex:
                traceback.print_exc()
                logging.error('\nException Caught While Running Chain\n{}'.format(traceback.format_exc()))
                raise ex

            self.chains, self.log_prob, self.acceptance_rate = data
            for key, value in self.chains.items():
                self.chains[key] = [value]

            self.log_prob = [self.log_prob]
        else:
            logging.info('\nRunning {} Chains in Parallel\nTotal: {}\nBurn: {}\nEvery: {}'
                         '\nStart: {}\nPrior Only: {}\nSeed: {}'.format(n_chains, total, burn, every,
                                                                        start, prior_only, seed))

            # generate a set of seeds to pass to each chain
            seeds = np.random.randint(2 ** 32 - 1, size=n_chains)
            processes = []
            vals = []
            Q = multiprocessing.Queue()  # Create a common queue to transfer the results

            for s in seeds:
                # Start N different processes each running the chain with a different seed
                # Each process needs a different seed or all the chains will run exactly alike

                p = multiprocessing.Process(target=self._run_chain_multi,
                                            args=(s, total, burn, every, start, prior_only, Q))
                processes.append(p)
                p.start()
            try:
                for _ in processes:  # Get all the results from each process. (Results have been pushed to the queue)
                    val = Q.get()
                    if isinstance(val, Exception):
                        raise val
                    vals.append(val)
            finally:
                for process in processes:  # Join all the processes
                    process.join()

            chains, log_prob, acceptance_rate = list(zip(*vals))
            tmp_chains = {key: [] for key in chains[0].keys()}
            for chain in chains:
                for key, value in tmp_chains.items():
                    value.append(chain[key])
            self.chains = tmp_chains
            self.acceptance_rate = np.mean(acceptance_rate)
            self.log_prob = np.array(log_prob)

    def get_chains(self, flattened=False, scaled=True):
        """
            Get the chains generated from run_chains()

            Each input variable will have a list of N numpy arrays where N is the number of chains run (controlled by
            n_chains in run_chains())

            :returns: A dictionary mapping input names to lists of numpy arrays containing the values generated from
                      running the MCMC algorithm
            :rtype: dict of (str, list of numpy array of float)
        """

        def lin_scaling(value, low, high, unscaled_low, unscaled_high):
            return (unscaled_high - unscaled_low) * (value - low) / (high - low) + unscaled_low

        def log_scaling(value, low, high, unscaled_low, unscaled_high):
            #return lin_scaling(np.log10(value), np.log10(low), np.log10(high), unscaled_low, unscaled_high)
            return lin_scaling(value, low, high, np.log10(unscaled_low), np.log10(unscaled_high))

        if flattened:
            val = {key: np.array(value).flatten() for key, value in self.chains.items()}
        else:
            val = {key: np.array(value) for key, value in self.chains.items()}

        if scaled is False:
            return val
        else:
            ret = {}
            for key, value in val.items():
                if key in self.inputs:
                    current_input = self.inputs[key]
                    scaling = current_input.scaling

                    if isinstance(scaling, str):
                        if current_input.unscaled_low is not None and current_input.unscaled_high is not None:
                            low, high, unscaled_low, unscaled_high, scaling = current_input.low, current_input.high, current_input.unscaled_low, current_input.unscaled_high, current_input.scaling
                            if scaling.lower()[:3] == 'lin':
                                ret[key] = lin_scaling(value, low, high, unscaled_low, unscaled_high)
                            elif scaling.lower()[:3] == 'log':
                                ret[key] = log_scaling(value, low, high, unscaled_low, unscaled_high)
                            else:
                                raise Exception('Unscaled Upper and Lower Bound must be specified.'
                                                'Upper Bound was {}. Lower Bound was {}'.format(unscaled_high, unscaled_low))
                    else:
                        ret[key] = value
                else:
                    ret[key] = value
            return ret

    def get_diagnostics(self, n_split=2, scaled=True):
        """
            Returns a set of diagnostics on the chains generated from run_chains()

            :parameter n_split: The number of sub-chains to divide each chain into
            :type n_split: int

            :returns: Dictionary with diagnostics:

                - ``rhat``: Measures convergence of chains. Closer to 1 is better. Should be less than 1.1
                - ``n_eff``: Measures the effective sample size. Higher is better. Should be greater that 10k
                - ``var_hat``: Estimates the variance of the samples.
                - ``autocorrelation``: Estimates the autocorrelation at lag t. Closer to 0 for low t is better.
                - ``mean``: The mean of the samples
                - ``std``: The standard deviation of the samples

            :rtype: dict of str
        """
        dt_diagnostics = {'acceptance_rate': self.acceptance_rate}
        #unscaled_chains = self.get_chains(scaled=scaled)
        unscaled_chains = self.get_chains(scaled=False)
        for key in unscaled_chains.keys():
            chains_ = list(map(functools.partial(mcmc_diagnostics.split, n_splits=n_split), unscaled_chains[key]))
            current_chains = []
            for chain_ in chains_:
                for sub_chain in chain_:
                    current_chains.append(sub_chain)
            dt_diagnostics[key] = {'r_hat'          : mcmc_diagnostics.r_hat(current_chains),
                                   'n_eff'          : mcmc_diagnostics.n_eff(current_chains),
                                   'var_hat'        : mcmc_diagnostics.var_hat(current_chains),
                                   'autocorrelation': {i: mcmc_diagnostics.rho_hat(current_chains, i) for i in
                                                       range(1, len(current_chains[0]))},
                                   'mean'           : np.mean(unscaled_chains[key]),
                                   'std'            : np.std(unscaled_chains[key]),
                                   'mode'           : mcmc_diagnostics.mode(current_chains)}

        return dt_diagnostics

    def diagnostics_string(self, n_split=2, autocorr=False, scaled=True):
        """
            Returns a set of diagnostics on the chains generated from run_chains()

            :parameter n_split: The number of sub-chains to divide each chain into
            :type n_split: int

            :parameter autocorr: Whether to include auto-correlation diagnostics
            :type autocorr: bool

            :returns: formated string of  diagnostics:

                - ``rhat``: Measures convergence of chains. Closer to 1 is better. Should be less than 1.1
                - ``n_eff``: Measures the effective sample size. Higher is better. Should be greater that 10k
                - ``var_hat``: Estimates the variance of the samples.
                - ``autocorrelation``: Estimates the autocorrelation at lag t. Closer to 0 for low t is better.
                - ``mean``: The mean of the samples
                - ``std``: The standard deviation of the samples

            :rtype: str
        """

        diags = self.get_diagnostics(n_split=n_split, scaled=scaled)

        strs = ['acceptance_rate:{}'.format(diags['acceptance_rate'])]
        for key, value in diags.items():
            if key is not 'acceptance_rate':
                strs.append('{}:'.format(key))
                for k, v in value.items():
                    if k is not 'autocorrelation':
                        strs.append('  {}:{}'.format(k, v))
                    elif autocorr:
                        strs.append('  {}:'.format(k))
                        for k1, v1 in v.items():
                            strs.append('    {}:{}'.format(k1, v1))

        return '\n'.join(strs)

    def get_residuals(self, scaled=True):
        """
            Returns the normalized residuals of the MCMC draws run through the surrogate models

            :returns: Normalized residuals
            :rtype: dict of list of float
        """

        def normalize(chains, output):
            residuals = []
            ls_input_vals = [chains[ipt] for ipt in output.inputs]
            for x_hat in zip(*ls_input_vals):
                x_hat = np.array(x_hat).T
                predicted = output.surrogate_model.predict(x_hat)
                residuals.append(
                    ((output.observed_value - predicted) / output.observed_std).flatten())

            return residuals

        return {key: normalize(self.get_chains(scaled=scaled), self.outputs[key]) for key in self.outputs.keys()}

    def trace_plot(self, input_name, ax=None, scaled=True):
        """
            Plots MCMC draws at each iteration number.

            :param input_name: Which input to plot
            :type input_name: str

            :param ax: The axes to plot to
            :type ax: matplotlib.axes.Axes
        """

        if ax is None:
            ax = plt
        for i, chain in enumerate(self.get_chains(scaled=scaled)[input_name]):
            ax.plot(i * len(chain) + np.arange(len(chain)), chain, linewidth=.1)
        if ax is plt:
            ax.show()

    def autocorr_plot(self, input_name, N=50, n_split=2, ax=None, scaled=True):
        """
            Plots auto-correlation values at given lag values.

            :param input_name: Which input to plot
            :type input_name: str

            :param N: The maximum lag value to plot
            :type N: int

            :param n_split: The number of chains to split each chain into when calculation auto-correlation
            :type n_split: int

            :param ax: The axes to plot to
            :type ax: matplotlib.axes.Axes
        """

        if ax is None:
            ax = plt
        autocorr = self.get_diagnostics(n_split, scaled=scaled)[input_name]['autocorrelation']
        X, Y = zip(*autocorr.items())
        ax.stem(X[:N], Y[:N], markerfmt='.')
        if ax is plt:
            ax.show()

    def residuals_plot(self, output_name, bins=10, ax=None, scaled=True):
        """
            Plots a histogram plot of the normalized residuals.

            :param output_name: Which output to plot
            :type output_name: str

            :param bins: The number of histogram bins to create
            :type bins: int

            :param ax: The axes to plot to
            :type ax: matplotlib.axes.Axes
        """

        if ax is None:
            fig, ax_tmp = plt.subplots(1, 1)
        else:
            ax_tmp = ax

        ax_tmp.axhline(3, color='xkcd:black', linestyle='--')
        ax_tmp.axhline(-3, color='xkcd:black', linestyle='--')
        for i, res in enumerate(self.get_residuals(scaled=scaled)[output_name]):
            ax_tmp.plot(i * len(res) + np.arange(len(res)), res, '.')

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax_tmp)
        ax_histogram = divider.append_axes('right', size='10%', pad='2%', sharey=ax_tmp)

        x_norm_pdf = np.linspace(-3, 3, 1000)
        y_norm_pdf = sts.norm().pdf(x_norm_pdf)

        ax_histogram.plot(y_norm_pdf, x_norm_pdf, color='xkcd:red')
        for res in self.get_residuals(scaled=scaled)[output_name]:
            ax_histogram.hist(res, bins=bins, alpha=.5, orientation='horizontal', density=True)
        if ax is None:
            plt.show()

    def posterior_predictive_plot(self, output_name, bins=10, ax=None):
        """
            Plots a histogram plot of the posterior predictive distribution for a given output.

            :param output_name: Which output to plot
            :type output_name: str

            :param bins: The number of histogram bins to create
            :type bins: int

            :param ax: The axes to plot to
            :type ax: matplotlib.axes.Axes
        """

        def predict(chains, output):
            ls_input_vals = [chains[ipt] for ipt in output.inputs]
            for x_hat in zip(*ls_input_vals):
                x_hat = np.array(x_hat).T
                pred = output.surrogate_model.predict(x_hat)
                yield pred

        if ax is None:
            ax = plt
        opt = self.outputs[output_name]
        ax.axvline(opt.observed_value, color='xkcd:black', linestyle='--')
        for prediction in predict(self.get_chains(scaled=False), opt):
            y_tilde = np.random.normal(prediction, opt.observed_std)
            ax.hist(y_tilde, bins=bins, alpha=.5, density=True)
        if ax is plt:
            ax.show()

    def log_posterior_plot(self, ax=None):
        """
            Plots log posterior probability of draws at each iteration number.

            :param ax: The axes to plot to
            :type ax: matplotlib.axes.Axes
        """
        if ax is None:
            ax = plt
        for i, ls in enumerate(self.log_prob):
            plt.plot(i * len(ls) + np.arange(len(ls)), ls, linewidth=.1)
        if ax is plt:
            ax.show()

    def histogram_plot(self, input_name, bins=10, density=True, alpha=.5, ax=None, scaled=True):
        """
            Plots a histogram plot of the posterior distribution for a given input.

            :param input_name: Which output to plot
            :type input_name: str

            :param bins: The number of histogram bins to create
            :type bins: int

            :param density: Whether to normalize the histogram such that it integrates to 1
            :type density: bool

            :param alpha: How opaque to make the plot. 0 is not visible. 1 is totally opaque.
            :type alpha: float

            :param ax: The axes to plot to
            :type ax: matplotlib.axes.Axes
        """

        if ax is None:
            ax = plt
        ax.hist(self.get_chains(flattened=True, scaled=scaled)[input_name], bins=bins, density=density, alpha=alpha)
        if ax is plt:
            ax.show()

    def _run_chain_multi(self, seed, total, burn, every, start, prior_only, queue):
        """
            A helper method for calling _run_chain with a queue

            Gets the value from _run_chain() and pushes it to the queue
        """
        try:
            val = self._run_chain(seed, total, burn, every, start, prior_only)
            queue.put(val)
        except Exception as ex:
            traceback.print_exc()
            logging.error('\nException Caught While Running Chain\n{}'.format(traceback.format_exc()))
            queue.put(ex)


class DefaultMCMC(MCMC):

    def _run_chain(self, seed, total, burn, every, start, prior_only=False):
        """
            Run a single MCMC chain

            Number of iterations = total*every + burn

            Iterates by asking _metropolis for a point. Only stores every Nth point where N is the 'every' parameter.

            :parameter seed: The seed for the chain
            :type seed: int

            :parameter total: The total number of sample points to return
            :type total: int

            :parameter burn: The number of burn-in iterations
            :type burn: int

            :parameter every: The rate at which to save points. Saves every Nth iteration
            :type every: int

            :parameter start: The value at which to start the chain
            :type start: dict of str, float

            :returns: **MCMC chain**: The values saved through the entire MCMC process
                      **acceptance rate**: The ratio between the number of accepted new points and total points
            :rtype: dict of str, numpy array of float
        """
        rng = np.random.RandomState(seed=seed)

        np_log_prob = np.zeros(total)

        start = self._init_start(start, rng)

        chain = {key: np.zeros(total) for key in start.keys()}

        accepted_total = 0
        i = 0
        j = 0
        current_point = start
        log_prob = None
        logging.info('Burning')
        print('Burning')
        while i < total:
            new_point, log_prob, accepted = self._metropolis(current_point, rng, prior_only, log_prob_current=log_prob)

            if j == burn + 1:
                logging.info('Sampling Start')
                print('Sampling Start')
                sys.stdout.flush()
            if j > burn and j % every == 0:
                for key in start.keys():
                    chain[key][i] = new_point[key]
                np_log_prob[i] = log_prob
                i += 1
                if i % (total / 10) == 0:
                    print("Sampling: {0:g}%".format((float(i) / float(total)) * 100))
                    sys.stdout.flush()
            j += 1

            accepted_total += accepted
            current_point = new_point

        logging.info("Sampling Done")
        print("Sampling Done")
        return chain, np_log_prob, float(accepted_total) / float(j)

    def _init_start(self, start, rng):
        if start is None:
            return {key: rng.uniform(self.inputs[key].low, self.inputs[key].high) for key in
                    self.inputs.keys()}
        else:
            return start

    def _metropolis(self, current_point, rng, prior_only=False, log_prob_current=None):
        """
            Accepts a new point or Rejects new point and keeps old point

            Proposes a new point and compares the ratio of the log probabilities of the new point and the old point to a
            uniformly drawn number. If the ratio is higher, the new point is accepted and returned. If the ratio is
            lower, the new point is rejected and the old point is returned.

            :parameter current_point: The current point of the markov chain
            :type current_point: dict of str, float

            :returns: **new point**: The new point if accepted or the old point if rejected
                      **accepted**: Whether the new point was accepted
            :rtype: (dict of str, float, bool)
        """
        proposed_point = self._propose(current_point, rng)

        log_prob_proposed = self._posterior_log_prob(proposed_point, prior_only)
        if log_prob_current is None:
            log_prob_current = self._posterior_log_prob(current_point, prior_only)

        f_alpha = np.exp(log_prob_proposed - log_prob_current)

        u = rng.uniform()
        if u < f_alpha:
            logging.debug('\nAccepted Point\nCurrent Point: {}\nNew Point: {}\n'
                          'Current Log Prob: {}\nNew Log Prob:'
                          '{}\nf alpha:{}\nu: {}'.format(current_point,
                                                         proposed_point,
                                                         log_prob_current,
                                                         log_prob_proposed,
                                                         f_alpha,
                                                         u))
            return proposed_point, log_prob_proposed, True
        else:
            logging.debug('\nRejected Point\nCurrent Point: {}\nNew Point: {}\n'
                          'Current Log Prob: {}\nNew Log Prob:'
                          '{}\nf alpha:{}\nu: {}'.format(current_point,
                                                         proposed_point,
                                                         log_prob_current,
                                                         log_prob_proposed,
                                                         f_alpha,
                                                         u))
            return current_point, log_prob_current, False

    def _propose(self, current_point, rng):
        """
            Propose a new point which is conditioned on the current point

            :parameter current_point: The current point which is the pre-state for the proposed point
            :type current_point: dict of (str, float)

            :returns: A new point whose distribution depends on the given current point
            :rtype: dict of (str, float)
        """
        return {key: rng.normal(current_point[key], self.inputs[key].proposal_sigma) for key in
                current_point.keys()}

    def _posterior_log_prob(self, current_point, prior_only=False):
        """
            Gets the log probability for a given point

            :parameter current_point: The given point to get the log probability for
            :type current_point: dict of (str, float)

            :returns: The log probability of the given point
        """

        num_outputs = len(self.outputs.keys())
        likelihood = 0.0
        y_vec = np.zeros(num_outputs)
        z_vec = np.zeros(num_outputs)
        sigma_y = np.zeros(num_outputs)
        sigma_z = np.zeros(num_outputs)

        if not prior_only:
            for idx, output in enumerate(self.outputs.values()):
                ls_input_vals = [current_point[ipt] for ipt in output.inputs]
                z_vec[idx], z_std = output.surrogate_model.predict(np.array([ls_input_vals]), return_std=True)
                y_vec[idx] = output.observed_value
                sigma_y[idx] = output.observed_std
                if z_std is not None:
                    sigma_z[idx] = z_std

            likelihood = sts.multivariate_normal(y_vec, np.diag(sigma_y ** 2) + np.diag(sigma_z ** 2)).logpdf(z_vec)

        def bounding_prior(input, x):
            return float('-inf') if x < input.low or x > input.high else input.prior(x)

        ls_prior = [bounding_prior(self.inputs[key], current_point[key]) for key in current_point.keys()]

        log_prob = likelihood + np.sum(ls_prior)
        logging.debug('\nCurrent Point: {}\nLog Prob: {}'.format(current_point, log_prob))
        return log_prob


class DiscrepancyMCMC(DefaultMCMC):

    def __init__(self):
        super(DiscrepancyMCMC, self).__init__()
        self.num_quant = None
        self.quantities = None
        self.tau_proposal_sigma = .05
        self.rho_proposal_sigma = .02
        self.tau_prior_alpha = 0.0
        self.tau_prior_beta = 0.0

    def _init_start(self, start, rng):
        quantities = []
        for output_name, output in self.outputs.items():
            if output.quantity not in quantities:
                quantities.append(output.quantity)

        self.num_quant = len(quantities)
        self.quantities = quantities

        if start is None:
            initial = {key: rng.uniform(self.inputs[key].low, self.inputs[key].high) for key in
                       self.inputs.keys()}

            tau = ['tau_{}'.format(i) for i in self.quantities]

            for t1 in tau:
                for t2 in tau:
                    first = t1.lstrip('tau_')
                    second = t2.lstrip('tau_')
                    if first != second and 'rho_{}_{}'.format(second, first) not in initial:
                        initial['rho_{}_{}'.format(first, second)] \
                            = 0.0 if self.rho_proposal_sigma == 0.0 else rng.uniform(-1, 1)

            for t in tau:
                initial[t] = sts.invgamma(1, 1).ppf(rng.uniform())
                #initial[t] = max(1.0e-6, rng.uniform(0.0, 3.0*self.tau_proposal_sigma)
        else:
            initial = deepcopy(start)

        return initial

    def _propose(self, current_point, rng):
        proposed_point = {
            key: rng.normal(current_point[key], self.inputs[key].proposal_sigma) if key in self.inputs else 0.0
            for key in current_point.keys()}

        for key in filter(lambda x: x.startswith('tau_'), current_point.keys()):
            proposed_point[key] = max(1e-6, rng.normal(current_point[key], self.tau_proposal_sigma))

        rps = self.rho_proposal_sigma
        for key in filter(lambda x: x.startswith('rho_'), current_point.keys()):
            proposed_point[key] \
                = 0.0 if rps == 0.0 else rng.normal(current_point[key], rps)

        return proposed_point

    def _posterior_log_prob(self, current_point, prior_only=False):
        num_outputs = len(self.outputs.keys())
        likelihood = 0.0
        tau_prob = np.zeros(self.num_quant)

        if not prior_only:
            z_vec = np.zeros(num_outputs)
            y_vec = np.zeros(num_outputs)
            sigma_delta = np.zeros((num_outputs, num_outputs))
            sigma_y = np.zeros(num_outputs)
            sigma_z = np.zeros(num_outputs)

            for idx, output in enumerate(self.outputs.values()):
                ls_input_vals = [current_point[ipt] for ipt in output.inputs]
                z_vec[idx], z_std = output.surrogate_model.predict(np.array([ls_input_vals]), return_std=True)
                y_vec[idx] = output.observed_value
                sigma_y[idx] = output.observed_std ** 2
                sigma_delta[idx, idx] = current_point['tau_{}'.format(output.quantity)] ** 2
                if z_std is not None:
                    sigma_z[idx] = z_std ** 2

                for jdx, output_alt in enumerate(self.outputs.values()):
                    if output.event == output_alt.event:
                        if idx != jdx:
                            rho1 = 'rho_{}_{}'.format(output.quantity, output_alt.quantity)
                            rho2 = 'rho_{}_{}'.format(output_alt.quantity, output.quantity)
                            if rho1 in current_point:
                                if not -1.0 <= current_point[rho1] <= 1.0:
                                    return float('-inf')
                                sigma_delta[idx, jdx] = current_point[rho1] * \
                                                        current_point['tau_{}'.format(output.quantity)] * \
                                                        current_point['tau_{}'.format(output_alt.quantity)]
                            elif rho2 in current_point:
                                if not -1.0 <= current_point[rho2] <= 1.0:
                                    return float('-inf')
                                sigma_delta[idx, jdx] = current_point[rho2] * \
                                                        current_point['tau_{}'.format(output.quantity)] * \
                                                        current_point['tau_{}'.format(output_alt.quantity)]

            try:
                likelihood = sts.multivariate_normal(mean=y_vec,
                                                     cov=np.linalg.multi_dot([np.diag(z_vec),
                                                                              sigma_delta,
                                                                              np.diag(z_vec)]) +
                                                         np.diag(sigma_y) + np.diag(sigma_z)).logpdf(z_vec)
            except (ValueError, np.linalg.LinAlgError):
                return float('-inf')

        for idx, quantity in enumerate(self.quantities):
            tau_prob[idx] = self._square_log_invgamma_proportional(current_point['tau_{}'.format(quantity)],
                                                                   self.tau_prior_alpha, self.tau_prior_beta)

        def bounding_prior(input, x):
            return float('-inf') if x < input.low or x > input.high else input.prior(x)

        ls_prior = [bounding_prior(self.inputs[key], current_point[key]) if key in self.inputs else 0 for key in
                    current_point.keys()]

        log_prob = likelihood + np.sum(ls_prior) + np.sum(tau_prob)

        logging.debug('\nCurrent Point: {}\nLog Prob: {}'.format(current_point, log_prob))

        return log_prob

    @staticmethod
    def _square_log_invgamma_proportional(x, alpha, beta):
        if x < 0:
            return float('-inf')
        return -(2 * alpha + 1) * np.log(x) - beta / (x ** 2)


class OutputVariable:
    """
        Represents an observed output
    """

    def __init__(self, event, quantity, surrogate_model, observed_value, observed_std, inputs):
        """
            :parameter name: The unique name of the output
            :type name: str

            :parameter surrogate_model: The surrogate model that represents the mapping of input values to output
                                        values. This model must be have a predict method that takes a numpy array and
                                        returns a numpy array (sklearn's fit/predict paradigm)

            :parameter observed_value: The observed experimental value
            :type observed_value: float

            :parameter observed_std: The error bound on the observed value
            :type observed_std: float

            :parameter inputs: The ordered names of the input variables that map to the output
            :type inputs: list of str
        """
        self.name = event + '_' + quantity
        self.surrogate_model = surrogate_model
        self.observed_value = observed_value
        self.observed_std = observed_std
        self.inputs = inputs
        self.event = event
        self.quantity = quantity


class InputVariable:
    """
        Represents an unobserved input
    """

    def __init__(self, name, low, high, proposal_sigma, prior=None, unscaled_low=None, unscaled_high=None, scaling=None):
        """

            :parameter name: The unique name of the input
            :type name: str

            :parameter low: The lower bound on the value this variable can take
            :type low: float

            :parameter high: The upper bound on the value this variable can take
            :type high: float

            :parameter proposal_sigma: The standard deviation of the proposal distribution for this variable
            :type proposal_sigma: float

            :parameter prior: The prior distribution of the variable.
                              If None, a uniform distribution over the whole range is used
            :type prior: function
        """
        self.name = name
        self.low = low
        self.high = high
        self.proposal_sigma = proposal_sigma
        if prior is None:
            self.prior = lambda x: 1.0
        else:
            self.prior = prior

        self.unscaled_low = unscaled_low
        self.unscaled_high = unscaled_high
        self.scaling = scaling


if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor

    logging.basicConfig(filename='foobar.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s\n')

    X = np.random.randn(100, 2) * 10
    Y = np.sqrt(np.power(X, 2).sum(axis=1).reshape(-1, 1))

    mdl = GaussianProcessRegressor().fit(X, Y)

    mdl2 = GaussianProcessRegressor().fit(X + 1.0, Y)

    exp1 = DefaultMCMC()

    exp1.tau_prior_beta = 0.0
    exp1.tau_prior_alpha = 0.0
    exp1.tau_proposal_sigma = 1.0
    exp1.rho_proposal_sigma = .5

    exp1.add_input('foo', -10, 10, 1.0, sts.norm.logpdf, unscaled_low=0, unscaled_high=1, scaling='log')

    exp1.add_input('bar', -10, 10, 1.0, sts.norm.logpdf, 1, 2)

    exp1.add_output('ax', 'height', mdl, 3, .4, ['foo', 'bar'])
    exp1.add_output('ay', 'height', mdl2, 3, .3, ['foo', 'bar'])
    exp1.add_output('ax', 'width', mdl, 3, .4, ['foo', 'bar'])
    exp1.add_output('ay', 'width', mdl2, 3, .3, ['foo', 'bar'])

    # exp1.run_chain(total=1000, burn=2000, every=50, n_chains=10, prior_only=True)
    # prior_points = exp1.get_chains(flattened=True)

    exp1.run_chain(total=1000, burn=200, every=5, n_chains=2)
    post_points = exp1.get_chains(flattened=True)

    print(exp1.diagnostics_string())
    logging.info('Diagnostics:\n' + exp1.diagnostics_string())

    # for ipt in exp1.get_chains().keys():
    #     fig, ax = plt.subplots(1, 1)
    #     exp1.trace_plot(ipt, ax=ax)
    #     fig.savefig('trace_plot_{}.png'.format(ipt))
    #     plt.close(fig)
    #
    #     fig, ax = plt.subplots(1, 1)
    #     exp1.autocorr_plot(ipt, ax=ax)
    #     fig.savefig('autocorr_plot_{}.png'.format(ipt))
    #     plt.close(fig)
    #
    #     fig, ax = plt.subplots(1, 1)
    #     exp1.histogram_plot(ipt, bins=100, density=True, ax=ax)
    #     fig.savefig('autocorr_plot_{}.png'.format(ipt))
    #     plt.close(fig)
    #
    # for opt in exp1.outputs.values():
    #     fig, ax = plt.subplots(1, 1)
    #     exp1.posterior_predictive_plot(opt.name, ax=ax)
    #     fig.savefig('posterior_predictive_plot_{}.png'.format(opt.name))
    #     plt.close(fig)
    #
    #     fig, ax = plt.subplots(1, 1)
    #     exp1.residuals_plot(opt.name, ax=ax)
    #     fig.savefig('residuals_plot_{}.png'.format(opt.name))
    #     plt.close(fig)
    #
    # fig, ax = plt.subplots(1, 2)
    # exp1.log_posterior_plot(ax=ax[0])
    exp1.histogram_plot('foo')
    exp1.histogram_plot('foo', scaled=False)
    # fig.savefig('log_posterior_plot1.png')
    # plt.close(fig)

    # plots.contour_plot(plt, post_points['foo'], post_points['bar'])
    # plt.show()
    # plots.scatter_plot(plt, post_points['foo'], post_points['bar'], 100)
    # plt.show()
