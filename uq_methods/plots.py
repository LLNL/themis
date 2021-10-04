from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy.ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable


def likelihood_plot(ax, prior_points, post_weights=None, post_points=None, exp_value=None,
                    conf_level=None, bins=10, density=True):
    """
        Shows the prior and posterior likelihood of a set of points.

        :parameter ax: The Axes to plot to
        :type ax: matplotlib axes object

        :parameter prior_points: A set of points
        :type prior_points: numpy array

        :parameter post_weights: The posterior weights of the points
        :type post_weights: numpy array

        :parameter exp_value: Experimental value
        :type exp_value: float

        :parameter conf_level: Confidence levels
        :type conf_level: float

        :parameter bins: Number of bins to partition into
        :type bins: int

        :parameter density: Whether to create a plot that integrates to 1
        :type density: bool
    """
    # if not len(prior) == len(posterior):
    #    raise Exception()

    if post_weights is not None and post_points is not None:
        raise Exception('Cannot use both weights and points at the same time')

    if post_weights is not None and np.any(post_weights < 0):
        raise Exception('Posterior weights cannot be negative')

    # Partition points into bins
    binned_prior, prior_x = np.histogram(prior_points, bins=bins, density=density)
    if post_weights is not None:
        binned_posterior, posterior_x = np.histogram(prior_points, bins=bins, weights=post_weights, density=density)
    elif post_points is not None:
        binned_posterior, posterior_x = np.histogram(post_points, bins=bins, density=density)

    prior_x = 0.5 * (prior_x[0:-1] + prior_x[1:])
    if post_weights is not None or post_points is not None:
        posterior_x = 0.5 * (posterior_x[0:-1] + posterior_x[1:])

    # Set y-axis limits
    max_prior = np.max(binned_prior)
    if post_weights is not None or post_points is not None:
        max_posterior = np.max(binned_posterior)
        upper_limit_y = max(max_prior, max_posterior) * 1.2
    else:
        upper_limit_y = max_prior * 1.2
    ax.set_ylim([0, upper_limit_y])

    # Set x-axis limits
    range_x = [np.min(prior_x), np.max(prior_x)]
    if exp_value is not None:
        lower_limit_x = min(exp_value - (.1 * (range_x[1] - exp_value)), range_x[0])
        upper_limit_x = max(exp_value + (.1 * (exp_value - range_x[0])), range_x[1])
    else:
        lower_limit_x = range_x[0]
        upper_limit_x = range_x[1]
    ax.set_xlim([lower_limit_x, upper_limit_x])

    # Plot distributions
    ax.fill_between(prior_x, binned_prior, color='blue', alpha=.25, label='Prior')
    if post_weights is not None or post_points is not None:
        ax.fill_between(posterior_x, binned_posterior, color='green', alpha=.25, label='Posterior')

    # Plot experimental value
    if exp_value is not None:
        ax.vlines(float(exp_value), 0, upper_limit_y, color='red', linestyle='dashed', label='Exp. Value')

    if conf_level is not None:
        if conf_level < 0 or conf_level > 100:
            raise Exception("Confidence level must be between 0 and 100")

        ls_alpha = [(100 - conf_level) / 2., conf_level / 2. + 50]

        # Plot confidence intervals
        ls_prior_index = _get_conf_level_index(binned_prior, ls_alpha)
        ax.plot(prior_x[ls_prior_index], binned_prior[ls_prior_index], color='blue', marker='o', linestyle='',
                label="Prior predictive {}% CI".format(conf_level))
        if post_weights is not None or post_points is not None:
            ls_posterior_index = _get_conf_level_index(binned_posterior, ls_alpha)
            ax.plot(posterior_x[ls_posterior_index], binned_posterior[ls_posterior_index], color='green', marker='o',
                    linestyle='',
                    label="Posterior predictive {}% CI".format(conf_level))
    ax.set_yticklabels([])
    ax.set_ylabel('density')

    if post_points is None and post_weights is None:
        return prior_x, binned_prior
    else:
        return prior_x, binned_prior, posterior_x, binned_posterior

def _get_conf_level_index(y, conf_levels):
    """
        Gives a the particular indices that given confidence levels occur at

        :parameter y: Set of points
        :type y: numpy array

        :parameter conf_levels: Confidence levels
        :type conf_levels: list of int

        :return: Indices of confidence levels
        :rtype: list of int
    """
    cdf = np.cumsum(y)
    ls_index = [np.searchsorted(cdf, 0.01 * l * cdf[-1], side='right') for l in conf_levels]

    # If the desired value ends up so that it should be inserted after the
    # last element in the list then we need to adjust the indices to prevent
    # index errors.
    num_cdf = len(cdf)
    ls_index = [i if i < num_cdf else (num_cdf - 1) for i in ls_index]
    return ls_index


def slice_plot_2d(ax, model, xvar, yvar, default, num_points=10, fig=None):
    """
        Shows a 2-D slice of a surrogate model

        :parameter ax: The Axes to plot to
        :type ax: matplotlib axes object

        :parameter model: Surrogate model to evaluate
        :type model: SurrogateModel

        :parameter xvar: The name of the variable to plot on x-axis
        :type xvar: str

        :parameter yvar: The name of the variable to plot on y-axis
        :type yvar: str

        :parameter default: the default point at which to make the slice
        :type default: numpy array

        :parameter num_points: The number of points in sliced axes to plot
        :type num_points: int

        Returns:  cax, X, Y, Z

    """
    x_names = model.X_names
    x_ranges = model.X_range

    xvar_index = np.where(x_names == xvar)[0][0]
    yvar_index = np.where(x_names == yvar)[0][0]

    xvar_range = x_ranges[xvar_index]
    yvar_range = x_ranges[yvar_index]

    xvar_points = np.linspace(xvar_range[0], xvar_range[1], num_points)
    yvar_points = np.linspace(yvar_range[0], yvar_range[1], num_points)

    X, Y = np.meshgrid(xvar_points, yvar_points)

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    points = np.repeat([default], num_points ** 2, 0).T
    points[xvar_index] = X_flat
    points[yvar_index] = Y_flat

    Z = model.predict(points.T)[0].reshape(num_points, num_points)

    cax = ax.contourf(X, Y, Z)
    ax.set_title("%s slice plot" % model.Y_names[0])
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

    if fig is not None:
        fig.colorbar(cax)
    return cax, X, Y, Z

def slice_plot_2d_std(ax, model, xvar, yvar, default, num_points=10, fig=None):
    """
        Shows a 2-D slice of a surrogate model's predicted standard deviation

        :parameter ax: The Axes to plot to
        :type ax: matplotlib axes object

        :parameter model: Surrogate model to evaluate
        :type model: SurrogateModel

        :parameter xvar: The name of the variable to plot on x-axis
        :type xvar: str

        :parameter yvar: The name of the variable to plot on y-axis
        :type yvar: str

        :parameter default: the default point at which to make the slice
        :type default: numpy array

        :parameter num_points: The number of points in sliced axes to plot
        :type num_points: int

        Returns: cax, X, Y, Z

    """
    x_names = model.X_names
    x_ranges = model.X_range

    xvar_index = np.where(x_names == xvar)[0][0]
    yvar_index = np.where(x_names == yvar)[0][0]

    xvar_range = x_ranges[xvar_index]
    yvar_range = x_ranges[yvar_index]

    xvar_points = np.linspace(xvar_range[0], xvar_range[1], num_points)
    yvar_points = np.linspace(yvar_range[0], yvar_range[1], num_points)

    X, Y = np.meshgrid(xvar_points, yvar_points)

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    points = np.repeat([default], num_points ** 2, 0).T
    points[xvar_index] = X_flat
    points[yvar_index] = Y_flat

    Z = model.predict(points.T, return_std=True)[1]

    if Z is None:
        raise ValueError("Surrogate Model did not return standard deviation")

    Z = Z.reshape(num_points, num_points)

    cax = ax.contourf(X, Y, Z)
    ax.set_title("%s slice plot [std]" % model.Y_names[0])
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

    if fig is not None:
        fig.colorbar(cax)

    return cax, X, Y, Z

def contour_plot(ax, x, y, bins=10, weights=None, fig=None):
    """
        Shows the shaded contours of a 2-D set of points, optionally weighted with posterior probabilities

        :parameter ax: The Axes to plot to
        :type ax: matplotlib axes object

        :parameter x: A set of points on the x-axis
        :type x: numpy array

        :parameter y: A set of points on the y-axis
        :type y: numpy array

        :parameter bins: Number of bins to partition into along each axis. Total number of bins will be N*N
        :type bins: int

        :parameter weights: The posterior weights of the points
        :type weights: numpy array
    
        Returns: cax, X, Y, Z

    """
    c_levels = [0.025, 0.25, 0.50, 0.75, 0.975]
    c_colors = ['blue', 'green', 'black', 'green', 'blue']
    c_styles = ['solid', 'dashed', 'solid', 'dashed', 'solid']

    Z, X, Y = np.histogram2d(x, y, bins=bins, weights=weights, density=True)
    X = 0.5 * (X[0:-1] + X[1:])
    Y = 0.5 * (Y[0:-1] + Y[1:])
    cax = ax.contourf(X, Y, Z.T)

    if fig is not None:
        fig.colorbar(cax)

    return cax, X, Y, Z

    # ax.contour(X, Y, Z,
    #           levels=c_levels,
    #           colors=c_colors,
    #           linestyles=c_styles)


def scatter_plot(ax, x, y, num_points=None, weights=None):
    """
        Shows the scatter plot of a 2-D set of points, optionally culled by posterior probabilities

        :parameter ax: The Axes to plot to
        :type ax: matplotlib axes object

        :parameter x: A set of points on the x-axis
        :type x: numpy array

        :parameter y: A set of points on the y-axis
        :type y: numpy array

        :parameter num_points: Number of points to plot
        :type num_points: int

        :parameter weights: The posterior weights of the points
        :type weights: numpy array
    """
    if num_points is None:
        ax.plot(x, y, 'o')
    else:
        if weights is not None:
            idx = np.random.choice(range(len(x)), num_points, p=weights/weights.sum())
        else:
            idx = np.random.choice(range(len(x)), num_points)
        ax.plot(x[idx], y[idx], '.')


def box_plot(ax, prior_preds, exp_obs, exp_std, posterior_wts=None, posterior_preds=None, add_uncert=None, num_pts=100,
             num_bins=100, seed=None):
    """
        Plots 3 box plots

        Shows 3 box plots representing the prior distribution, posterior distribution,
        and posterior distribution + additional uncertainties. Also shows a histogram of
        posterior distribution + additional uncertainties.

        :parameter ax: The Axes to plot to
        :type ax: matplotlib axes object

        :parameter prior_preds: A set of points
        :type prior_preds: numpy array

        :parameter posterior_wts: The posterior weights of the points
        :type posterior_wts: numpy array

        :parameter exp_obs: Experimental observed value
        :type exp_obs: float

        :parameter exp_std: Experimental standard deviation
        :type exp_std: float

        :parameter add_uncert: Additional uncertainties
        :type add_uncert: list of 2-tuple of float

        :parameter num_pts: Number of points to sample from probability models
        :type num_pts: int

        :parameter num_bins: Number of histogram bins
        :type num_bins: int

        :parameter seed: Random seed
        :type seed: int
    """

    def model_pred_dists(points, weights, std=None, bins=10, std_mltpl=10, add_uncert=None):
        """
            Convolution step

            :parameter points: A set of points
            :type points: numpy array

            :parameter weights: A set of posterior weights
            :type weights: numpy array

            :parameter std: Experimental standard deviation
            :type std: float

            :parameter bins: Number of histogram bins
            :type bins: int

            :parameter std_mltpl: Standard deviation multiplier
            :type std_mltpl: float

            :parameter add_uncert: Additional uncertainties
            :type add_uncert: list of 2-tuple of float

            :return: A set of edges and heights. Optionally a set of heights associated with additional uncertainties
            :rtype: numpy array, numpy array, list of numpy array

            Returns: mean and std for model_prior, model_post and pred_post

        """
        if add_uncert is None:
            add_uncert = []

        mu = np.mean(points)
        low = mu - std_mltpl * std
        upp = mu + std_mltpl * std
        gauss_filter = scipy.ndimage.gaussian_filter1d

        heights, edges = np.histogram(points, bins=bins, density=True, range=[low, upp], weights=weights)
        dy = (edges[-1] - edges[0]) / bins

        ls_heights = [heights]
        for _, f_std in add_uncert:
            ls_heights.append(gauss_filter(ls_heights[-1], (mu * f_std / 100.0) / dy, order=0, mode='constant'))

        return edges, heights, ls_heights

    if posterior_wts is not None and posterior_preds is not None:
        raise Exception('Cannot use both weights and points at the same time')
    elif posterior_wts is None and posterior_preds is None:
        raise Exception('Must specify one of posterior_wts or posterior_preds')

    if add_uncert is None:
        add_uncert = []

    prior_edges, mod_prior_heights, pred_prior_heights = model_pred_dists(prior_preds,
                                                                          np.ones(prior_preds.shape),
                                                                          std=exp_std,
                                                                          bins=num_bins)
    dy = prior_edges[1] - prior_edges[0]

    if seed is not None:
        np.random.seed(seed + 1)
    np_model_prior = np.random.choice(prior_edges[0:-1] + 0.5 * dy, num_pts,
                                      p=mod_prior_heights / mod_prior_heights.sum())
    model_prior_mean = np.mean(np_model_prior)
    model_prior_std = np.std(np_model_prior)

    if posterior_wts is not None:
        post_edges, mod_post_heights, ls_pred_post_heights = model_pred_dists(prior_preds,
                                                                              posterior_wts,
                                                                              std=exp_std,
                                                                              bins=num_bins,
                                                                              add_uncert=add_uncert)
    elif posterior_preds is not None:
        post_edges, mod_post_heights, ls_pred_post_heights = model_pred_dists(posterior_preds,
                                                                              np.ones(posterior_preds.shape),
                                                                              std=exp_std,
                                                                              bins=num_bins,
                                                                              add_uncert=add_uncert)

    pred_post_heights = ls_pred_post_heights[-1]

    dy = post_edges[1] - post_edges[0]

    if seed is not None:
        np.random.seed(seed + 2)
    np_model_post = np.random.choice(post_edges[0:-1] + 0.5 * dy, num_pts, p=mod_post_heights / mod_post_heights.sum())
    model_post_mean = np.mean(np_model_post)
    model_post_std = np.std(np_model_post)

    np_pred_post = np.random.choice(post_edges[0:-1] + 0.5 * dy, num_pts,
                                    p=pred_post_heights / pred_post_heights.sum())
    pred_post_mean = np.mean(np_pred_post)
    pred_post_std = np.std(np_pred_post)

    ls_dists = [np_model_prior, np_model_post, np_pred_post]

    ls_x_lbl = ['prior\npredictive',
                'posterior\npredictive',
                'posterior\npredictive\n+ addl_uncert\n({})'.format(
                    '\n+ '.join(['{}'.format(tpUnc[0]) for tpUnc in add_uncert]))]
    pos = list(range(1, len(ls_dists) + 1))

    if model_prior_mean > 1.0:
        ax.annotate("$\mu$={:.2f}\n$\sigma$={:.2f}".format(model_prior_mean, model_prior_std),
                    (.05, .8), xycoords='axes fraction')
        ax.annotate("$\mu$={:.2f}\n$\sigma$={:.2f}".format(model_post_mean, model_post_std),
                    (.35, .8), xycoords='axes fraction')
        ax.annotate("$\mu$={:.2f}\n$\sigma$={:.2f}".format(pred_post_mean, pred_post_std),
                    (.65, .8), xycoords='axes fraction')
    else:
        ax.annotate("$\mu$={:.4f}\n$\sigma$={:.4f}".format(model_prior_mean, model_prior_std),
                    (.05, .8), xycoords='axes fraction')
        ax.annotate("$\mu$={:.4f}\n$\sigma$={:.4f}".format(model_post_mean, model_post_std),
                    (.35, .8), xycoords='axes fraction')
        ax.annotate("$\mu$={:.4f}\n$\sigma$={:.4f}".format(pred_post_mean, pred_post_std),
                    (.65, .8), xycoords='axes fraction')

    ax.boxplot(ls_dists, sym='', whis=1.5, patch_artist=True, positions=pos, medianprops={'color': 'k'})
    ax.set_xticklabels(ls_x_lbl, rotation=30)

    divider = make_axes_locatable(ax)
    ax_histogram = divider.append_axes('right', size='10%', pad='2%', sharey=ax)
    yy = np.ravel(list(zip(post_edges, post_edges)))
    xx = np.ravel(list(zip([0.] + list(pred_post_heights), list(pred_post_heights) + [0.])))
    ax_histogram.fill_betweenx(yy, xx)

    return (model_prior_mean, model_prior_std,
            model_post_mean, model_post_std,
            pred_post_mean, pred_post_std)
