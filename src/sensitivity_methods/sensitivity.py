from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import lars_path
from sklearn.preprocessing import PolynomialFeatures

from sampling_methods.sampler import LatinHyperCubeSampler
from sensitivity_methods.pce_model import PolynomialChaosExpansionModel


def _variance_network_plot(ax, feature_data, response_data, feature_names, response_names, score_function, method_label,
                           degree=2, max_size=20.0, alpha=.5, label_size=12, **kwargs):
    """

        Create a set of network plots based on a given set of data and score function
        for each combination of output and degree.

        Network plots compare scores between different degree of interaction.
        In all plots, each parameter by itself is represented as a node in a graph.
        For plots of degree 2, interactions between 2 parameters are represented
        as an edge between the respective nodes.
        For plots of degree 3 or higher, interactions between the respective parameters
        are represented as a hyper edge i.e. an edge between 3 or more nodes.
        The sizes and thicknesses of the nodes and edges correspond to the scores from the given score function.

        Args:
            - ax ([[matplotlib.Axes]]): Array-like of Axes to plot to. Dimension is number of outputs by (degree-1)
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - score_function (function): Function which scores features based on importance to responses
            - method_label (string): Label of method used
            - degree: maximum degree of interactions to plot
            - max_size (float): Maximum size of elements in plot. Measured in points
            - alpha (float): Opacity of elements in plot.
            - label_size (int): Font size of labels. Measured in points
            - kwargs: Keyword arguments to be passed to score_function

        Raises:
            -
    """
    assert degree >= 2

    feature_interaction_data, feature_interaction_names, powers = _make_interactions(feature_data, feature_names,
                                                                                     degree=degree,
                                                                                     interaction_only=True)
    feature_interaction_names = np.array(feature_interaction_names)
    powers_degree = powers.sum(axis=1)

    for response_column_data, response_column_name, plot_column in zip(response_data.T, response_names, ax.T):

        scores = score_function(feature_interaction_data, response_column_data, powers=powers, **kwargs)

        node_weights = scores[powers_degree == 1]

        node_names = np.array(feature_names)
        node_labels = {name: "{}\n{:.2e}".format(name,
                                                 weight) for name, weight in zip(node_names,
                                                                                 node_weights)}

        for degree_to_plot, plot_row in zip(range(2, degree + 1), plot_column):
            edge_weights = scores[powers_degree == degree_to_plot]

            size_factor = max_size / max(node_weights.max(), edge_weights.max())

            node_sizes = size_factor * node_weights
            edge_sizes = size_factor * edge_weights

            edge_list = [tuple(node_names[p]) for p in powers[powers_degree == degree_to_plot].astype('bool')]
            edge_names = feature_interaction_names[powers_degree == degree_to_plot]
            edge_labels = {names: "{}\n{:.2e}".format(int_name,
                                                      weight) for names, int_name, weight in zip(edge_list,
                                                                                                 edge_names,
                                                                                                 edge_weights)}

            if degree_to_plot == 2:
                # normal graph
                graph = nx.complete_graph(node_names)

                pos = nx.circular_layout(graph, scale=1.0)

                nx.draw_networkx_nodes(graph, pos,
                                       node_size=node_sizes ** 2,
                                       node_color='xkcd:red',
                                       linewidths=1.0,
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_labels(graph, pos,
                                        labels=node_labels,
                                        font_size=label_size,
                                        ax=plot_row)

                nx.draw_networkx_edges(graph, pos,
                                       width=edge_sizes,
                                       edge_color='xkcd:green',
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_edge_labels(graph, pos,
                                             edge_labels=edge_labels,
                                             font_size=label_size,
                                             label_pos=.6,
                                             ax=plot_row)

            else:
                # hyper graph
                graph = nx.Graph()
                graph.add_nodes_from(node_names)
                graph.add_nodes_from(edge_list)
                for edge, weight in zip(edge_list, edge_weights):
                    for node in edge:
                        graph.add_edge(node, edge, weight=weight)

                hyper_edge_weights = [size_factor * el['weight'] for el in graph.edges.values()]

                pos = nx.bipartite_layout(graph, edge_list, align='horizontal')

                nx.draw_networkx_nodes(graph, pos,
                                       nodelist=node_names,
                                       node_size=node_sizes ** 2,
                                       node_color='xkcd:red',
                                       linewidths=1.0,
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_nodes(graph, pos,
                                       nodelist=edge_list,
                                       node_size=edge_sizes ** 2,
                                       node_color='xkcd:green',
                                       linewidths=1.0,
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_labels(graph, pos,
                                        labels=node_labels,
                                        font_size=label_size,
                                        ax=plot_row)

                nx.draw_networkx_labels(graph, pos,
                                        labels=edge_labels,
                                        font_size=label_size,
                                        ax=plot_row)

                nx.draw_networkx_edges(graph, pos,
                                       edge_color='xkcd:green',
                                       width=hyper_edge_weights,
                                       alpha=alpha,
                                       ax=plot_row)

            plot_row.set_title("'{}' degree {} ({})".format(response_column_name, degree_to_plot, method_label))


def _rank_plot(ax, feature_data, response_data, feature_names, response_names, score_function,
               degree=1, interaction_only=True, **kwargs):
    """
        Create a rank plot based on a given set of data and score function.

        Ranks each feature from 1 to N based on the result of score_function.
        A grid of boxes are plotted with response names on the x-axis
        and feature names (and possibly interactions) on the y-axis.
        Each box is colored and labeled according to its calculated rank.
        Ranks are only calculated within responses, not across responses.

        Args:
            - ax (matplotlib.Axes): Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - score_function (function): Function which scores features based on importance to responses
            - degree (int): Maximum degree of interaction
            - interaction_only (bool): Whether to only include lowest powers of interaction or include higher powers
            - kwargs: Keyword arguments passed to score_function

        Raises:
            -

    """

    feature_interaction_data, feature_interaction_names, powers = _make_interactions(feature_data, feature_names,
                                                                                     degree=degree,
                                                                                     interaction_only=interaction_only)

    num_features = feature_interaction_data.shape[1]
    num_responses = response_data.shape[1]

    scores = np.row_stack([score_function(feature_interaction_data,
                                          response_column,
                                          powers=powers, **kwargs) for response_column in response_data.T])

    order = np.argsort(-scores)
    rank_table = np.argsort(order)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    color_mesh = ax.pcolormesh(num_features - rank_table.T, edgecolors='None', lw=0.0, cmap=cm.plasma)

    ax.hlines(np.arange(1, num_features + 1), 0, num_responses, color='k')
    ax.vlines(np.arange(1, num_responses + 1), 0, num_features, color='k')

    cbar = plt.colorbar(color_mesh, cax=cax, orientation='vertical', values=np.arange(num_features) + 1)

    for i in range(num_responses * num_features):
        idx = np.unravel_index(i, (num_responses, num_features))
        ax.text(idx[0] + .5, idx[1] + .5, rank_table[idx] + 1, ha='center', va='center')

    ax.set_xticks(np.arange(num_responses) + 0.5)
    ax.set_xticklabels(response_names, rotation=35, ha='right')
    ax.set_xlabel('Outputs', ha='center')

    ax.set_yticks(np.arange(num_features) + 0.5)
    ax.set_yticklabels(feature_interaction_names)
    ax.set_ylabel('Parameters')

    cbar_labels = np.arange(1, num_features + 1).astype('str')
    cbar_labels[-1] += ' (Least Sensitive)'
    cbar_labels[0] += ' (Most Sensitive)'

    cbar.set_ticks(num_features - np.arange(num_features))
    cbar.set_ticklabels(cbar_labels)

    ax.set_xlim([0, num_responses])
    ax.set_ylim([0, num_features])
    ax.invert_yaxis()


def _score_plot(ax, feature_data, response_data, feature_names, response_names, score_function, title,
                degree=1, interaction_only=True, y_axis_label='Score', **kwargs):
    """
        Create a set of bar plots based on a given set of data and score function for each output.

        Uses score_function to assign a score to each feature or interaction of features.
        These scores are then plotted as a bar plot, with the height of each bar corresponding to the calculated score.


        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - score_function (function): Function which scores features based on importance to responses
            - title (string): Title of plot
            - degree (int): Maximum degree of interaction
            - interaction_only (bool): Whether to only include lowest powers of interaction or include higher powers
            - y_axis_label (string): Y axis label
            - kwargs: Keyword arguments passed to score_function

        Raises:
            -
    """

    feature_interaction_data, feature_interaction_names, powers = _make_interactions(feature_data, feature_names,
                                                                                     degree=degree,
                                                                                     interaction_only=interaction_only)

    for y, y_name, axes in zip(response_data.T, response_names, ax):
        scores = score_function(feature_interaction_data, y, powers=powers, **kwargs)

        axes.bar(feature_interaction_names, scores)
        axes.tick_params(axis='x', labelrotation=70)
        axes.set_xlabel('Parameter Interaction')
        axes.set_ylabel(y_axis_label)
        axes.set_title('{} ({})'.format(title, y_name))


def _make_interactions(feature_data, feature_names, degree=2, interaction_only=False):
    """
        Create interactions between features

        Creates the polynomial interactions by creating all possible combinations between each individual feature.
        An interaction data column is the product of 2 or more features.
        An interaction name is a concatenation of the names of the columns used to create the interaction.
        If interaction_only is False, then feature columns are allowed to interact with themselves.
        In other word, if interaction_only is False,
        then feature columns may be raised to a power in their interactions.

        Args:
            - feature_data ([[float]]): Array-like of feature data
            - feature_names ([string]): Array-like of feature names
            - degree (int): Maximum degree of interaction
            - interaction_only (bool): Whether to only include lowest powers of interaction or include higher powers

        Returns:
             - Feature interaction data ([[float]]): The resulting feature data from the interactions
             - Feature interaction names ([string]): The resulting feature names from the interactions
             - Feature interaction powers ([[int]]): An array containing the powers to which each feature was raised
                                                     to create the interactions.

        Raises:
            -
    """

    feature_interaction_names = []

    if interaction_only:
        transformer = PolynomialFeatures(degree=degree,
                                         interaction_only=True,
                                         include_bias=False)

        for deg in range(1, degree + 1):
            feature_interaction_names.extend([':'.join(ls) for ls in itertools.combinations(feature_names, r=deg)])
    else:
        transformer = PolynomialFeatures(degree=degree,
                                         interaction_only=False,
                                         include_bias=False)
        for deg in range(1, degree + 1):
            feature_interaction_names.extend(
                [':'.join(ls) for ls in itertools.combinations_with_replacement(feature_names, r=deg)])

    transformer.fit(feature_data)
    return transformer.transform(feature_data), feature_interaction_names, transformer.powers_


def _f_score(feature_data, response_data, center=True, **kwargs):
    """
        Scores features based on an F-test of linear regression coefficients

        Each features is scored using an F-test of linear regression coefficients.
        The F statistic is calculated by looking at the difference between a linear model with
        and without the feature included.
        See sklearn.feature_selction.f_regression for more information.

        Args:
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - center (bool): Whether to center data feature_data before calculating scores
            - kwargs: Captures superfluous keyword arguments

        Returns:
            - Scores ([float]): The scores of each feature

        Raises:
            -
    """
    scores, p_values = f_regression(feature_data, response_data, center=center)
    return scores


def _f_p_score(feature_data, response_data, center=True, **kwargs):
    """
        Give the p-value for an F-test of linear regression coefficients for each feature.

        Each features is scored using an F-test of linear regression coefficients.
        The F statistic is calculated by looking at the difference between a linear model with
        and without the feature included.
        The p-value of this F statistic is returned.
        In general, a p-value < 0.05 is considered significant, meaning that particular feature is worth keeping.
        See sklearn.feature_selction.f_regression for more information.

        Args:
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - center (bool): Whether to center data feature_data before calculating scores
            - kwargs: Captures superfluous keyword arguments

        Returns:
            - p_values ([float]): The p-values of each feature

        Raises:
            -
    """
    scores, p_values = f_regression(feature_data, response_data, center=center)
    return p_values


def _mutual_info_score(feature_data, response_data, n_neighbors=3, **kwargs):
    """
        Scores each feature based on the amount of mutual information shared with the response.

        Each feature is scored using an estimate of mutual information between it and the response data.
        Mutual information measures the amount of information that can be obtained about the response
        by observing the particular feature.
        See sklearn.eature_selection.mutual_info_regression for more information.

        Args:
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - n_neighbors (int): How many neighboring bins to consider when estimating mutual information
            - kwargs: Captures superfluous keyword arguments

        Returns:
            - Scores ([float]): The scores of each feature

        Raises:
            -
    """
    return mutual_info_regression(feature_data, response_data, n_neighbors=n_neighbors)


def _pce_score(feature_data, response_data, ranges, powers, pce_degree=1, model_degrees=1, **kwargs):
    """
        Scores each feature based on the amount of variance it contributes to the response in a PCE model.

        Each feature is scored using variance decomposition with a Polynomial Chaos Expansion (PCE) model.
        The PCE model uses linear regression on an basis of orthogonal polynomials.
        Using the coefficients of this model,
        each feature's contribution the the variance of the response can be assigned.
        The score is the portion of response variance that the feature is estimated to have contributed.

        Args:
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - ranges ([[float]]): Array-like of feature ranges.
                                  Each row is a length 2 array of the lower and upper bounds.
            - pce_degree (int): Maximum degree of interaction to score
            - model_degrees (int): Maximum degree of interaction for PCE model
            - kwargs: Captures superfluous keyword arguments

        Returns:
            - Scores ([float]): The scores of each feature

        Raises:
            -
    """
    feature_data_to_fit = feature_data[:, powers.sum(axis=-1) == 1]

    model = PolynomialChaosExpansionModel(num_degrees=model_degrees, ranges=ranges)
    model.fit(feature_data_to_fit, response_data[:, np.newaxis])

    return model.make_contributions(pce_degree).reshape(-1)


def one_at_a_time_effects(feature_data, response_data):
    m, n = feature_data.shape
    assert m == 2*n+1

    diff_features = feature_data[0] - feature_data
    diff_response = response_data[0] - response_data

    which_var = diff_features != 0

    val = diff_response[1:]/diff_features[which_var]

    return val.reshape(-1, 2).T


def morris_effects(feature_data, response_data):
    n, k = feature_data.shape
    r = int(n / (k + 1))

    def make_effect(feature_partition, response_partition):
        diff_response = response_partition[1:] - response_partition[:-1]
        diff_features = feature_partition[1:] - feature_partition[:-1]
        which_var = diff_features != 0
        val = diff_response / diff_features[which_var]
        return np.dot(val, which_var.astype('int'))

    return np.array([make_effect(_x, _y) for _x, _y in zip(feature_data.reshape(r, k + 1, k),
                                                           response_data.reshape(r, k + 1))])


def lasso_path_plot(ax, feature_data, response_data, feature_names, response_names, degree=1, method='lasso'):
    """
        Plots Lasso paths

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - degree (int): Maximum degree of interaction
            - method (string): Which algorithm to use; lasso: Coordinate descent, lars: least angle regression

        Raises:
            -
    """
    feature_interaction_data, feature_interaction_names, _ = _make_interactions(feature_data, feature_names,
                                                                                degree=degree, interaction_only=False)

    for response_column_data, response_column_name, axes in zip(response_data.T, response_names, ax):

        alphas, active, coefficients = lars_path(feature_interaction_data, response_column_data, method=method)

        shrinkage = coefficients.T.abs().sum(axis=1)
        shrinkage /= shrinkage[-1]

        lines = [axes.plot(shrinkage, coefficient, color=np.random.rand(3))[0] for coefficient in coefficients]
        axes.legend(lines, feature_interaction_names)

        for s in shrinkage:
            axes.axvline(s, 0, 1, linestyle='dashed', color='grey')
        axes.hlines(0, 0, 1, linestyles='dashed', color='grey')

        axes.set_xlabel('Shrinkage Factor')
        axes.set_ylabel('Coefficient')
        axes.set_title('{} Path'.format(method.upper()))


def sensitivity_plot(ax, surrogate_model, feature_names, response_names, feature_ranges,
                     num_plot_points=100, num_seed_points=5, seed=2018):
    """
        Plots sensitivity plots

        Args:
            - ax ([[matplotlib.Axes]]): Array-like of Axes to plot to
            - surrogate_model (surrogate_model): Surrogate model which has been fit
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - feature_ranges ([[float]]): Array-like of feature ranges.
                                          Each row is a length 2 array of the lower and upper bounds.
            - num_plot_points (int): Number of points to plot on each dimension sweep
            - num_seed_points (int): Number of points to use as default points
            - seed (int): RNG seed

        Raises:
            -
    """
    rand_gen = np.random.default_rng(seed)

    np_seed_points = LatinHyperCubeSampler.sample_points(num_points=num_seed_points,
                                                         box=feature_ranges,
                                                         seed=seed)
    colors = rand_gen.random((num_seed_points, 3), dtype='float')

    for color, point in zip(colors, np_seed_points):
        for feature_index, feature_name in enumerate(feature_names):

            dimension_sweep = np.linspace(feature_ranges[feature_index][0],
                                          feature_ranges[feature_index][1],
                                          num_plot_points)
            new_feature_data = np.tile(point, reps=(num_plot_points, 1))
            new_feature_data[:, feature_index] = dimension_sweep
            response_prediction = surrogate_model.predict(new_feature_data)

            for response_index, response_name in enumerate(response_names):
                ax[feature_index][response_index].set_xlabel(feature_name)
                ax[feature_index][response_index].set_ylabel(response_name)
                ax[feature_index][response_index].plot(dimension_sweep, response_prediction[:, response_index],
                                                       color=color, alpha=.75)


def f_score_plot(ax, feature_data, response_data, feature_names, response_names,
                 degree=1, interaction_only=True, use_p_value=False):
    """
        Plots F score plot

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - degree (int): Maximum degree of interaction
            - interaction_only (bool): Whether to only include lowest powers of interaction or include higher powers
            - use_p_value (bool): Whether to use p-values or raw F-score

        Raises:
            -
    """
    _score_plot(ax=ax,
                feature_data=feature_data,
                response_data=response_data,
                feature_names=feature_names,
                response_names=response_names,
                degree=degree,
                interaction_only=interaction_only,
                score_function=_f_p_score if use_p_value else _f_score,
                title='F Score',
                y_axis_label='p-value' if use_p_value else 'Score')

    if use_p_value:
        for axes in ax:
            axes.axhline(.05, 0, 1, linestyle='dashed', color='red', label='alpha=.05')


def mutual_info_score_plot(ax, feature_data, response_data, feature_names, response_names, n_neighbors=3):
    """
        Plots mutual information score plot

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - n_neighbors (int): How many neighboring bins to consider when estimating mutual information

        Raises:
            -
    """
    _score_plot(ax=ax,
                feature_data=feature_data,
                response_data=response_data,
                feature_names=feature_names,
                response_names=response_names,
                degree=1,
                interaction_only=False,
                score_function=_mutual_info_score,
                title='Mutual Information',
                y_axis_label='Shared Information (nats)',
                n_neighbors=n_neighbors)


def pce_score_plot(ax, feature_data, response_data, feature_names, response_names, feature_ranges,
                   degree=1, model_degrees=1):
    """
        Plots PCE score plot

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - feature_ranges ([[float]]): Array-like of feature ranges.
                                          Each row is a length 2 array of the lower and upper bounds.
            - degree (int): Maximum degree of interaction to plot
            - model_degrees (int): Maximum degree of interaction for PCE model

        Raises:
            -
    """
    _score_plot(ax=ax,
                feature_data=feature_data,
                response_data=response_data,
                feature_names=feature_names,
                response_names=response_names,
                degree=degree,
                interaction_only=True,
                score_function=_pce_score,
                title='PCE Variance Decomposition',
                y_axis_label='Variance Contribution',
                ranges=feature_ranges,
                pce_degree=degree,
                model_degrees=model_degrees)


def f_score_rank_plot(ax, feature_data, response_data, feature_names, response_names,
                      degree=1, interaction_only=True, use_p_value=False):
    """
        Plots F score rank plot

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - degree (int): Maximum degree of interaction
            - interaction_only (bool): Whether to only include lowest powers of interaction or include higher powers
            - use_p_value (bool): Whether to use p-values or raw F-score

        Raises:
            -
    """
    _rank_plot(ax=ax,
               feature_data=feature_data,
               response_data=response_data,
               feature_names=feature_names,
               response_names=response_names,
               score_function=(lambda *args: -_f_p_score(*args)) if use_p_value else _f_score,
               degree=degree,
               interaction_only=interaction_only)


def mutual_info_rank_plot(ax, feature_data, response_data, feature_names, response_names, n_neighbors=3):
    """
        Plots mutual information rank plot

        Args:
            - ax (matplotlib.Axes): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - n_neighbors (int): How many neighboring bins to consider when estimating mutual information

        Raises:
            -
    """
    _rank_plot(ax=ax,
               feature_data=feature_data,
               response_data=response_data,
               feature_names=feature_names,
               response_names=response_names,
               score_function=_mutual_info_score,
               degree=1,
               interaction_only=False,
               n_neighbors=n_neighbors)


def pce_rank_plot(ax, feature_data, response_data, feature_names, response_names, feature_ranges,
                  degree=1, model_degrees=1):
    """
        Plots PCE rank plot

        Args:
            - ax (matplotlib.Axes): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - feature_ranges ([[float]]): Array-like of feature ranges.
                                          Each row is a length 2 array of the lower and upper bounds.
            - degree (int): Maximum degree of interaction to plot
            - model_degrees (int): Maximum degree of interaction for PCE model

        Raises:
            -
    """
    _rank_plot(ax=ax,
               feature_data=feature_data,
               response_data=response_data,
               feature_names=feature_names,
               response_names=response_names,
               score_function=_pce_score,
               degree=degree,
               interaction_only=True,
               ranges=feature_ranges,
               pce_degree=degree,
               model_degrees=model_degrees)


def f_score_network_plot(ax, feature_data, response_data, feature_names, response_names,
                         degree=2, max_size=10.0, label_size=10, alpha=.5):
    """
        Plots F score network plot

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - degree (int): Maximum degree of interaction
            - max_size (float): Maximum size of elements in plot. Measured in points
            - label_size (int): Font size of labels. Measured in points
            - alpha (float): Opacity of elements in plot.

        Raises:
            -
    """
    _variance_network_plot(ax=ax,
                           feature_data=feature_data,
                           response_data=response_data,
                           feature_names=feature_names,
                           response_names=response_names,
                           score_function=_f_score,
                           method_label='F score',
                           degree=degree,
                           max_size=max_size,
                           label_size=label_size,
                           alpha=alpha)


def pce_network_plot(ax, feature_data, response_data, feature_names, response_names, feature_ranges,
                     degree=2, model_degrees=2, max_size=10.0, label_size=10, alpha=.5):
    """
        Plots PCE network plot

        Args:
            - ax (matplotlib.Axes): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row is an observation
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature data
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - feature_ranges ([[float]]): Array-like of feature ranges.
                                          Each row is a length 2 array of the lower and upper bounds.
            - degree (int): Maximum degree of interaction to plot
            - model_degrees (int): Maximum degree of interaction for PCE model
            - max_size (float): Maximum size of elements in plot. Measured in points
            - label_size (int): Font size of labels. Measured in points
            - alpha (float): Opacity of elements in plot.

        Raises:
            -
    """
    _variance_network_plot(ax=ax,
                           feature_data=feature_data,
                           response_data=response_data,
                           feature_names=feature_names,
                           response_names=response_names,
                           score_function=_pce_score,
                           method_label='PCE',
                           degree=degree,
                           max_size=max_size,
                           label_size=label_size,
                           alpha=alpha,
                           ranges=feature_ranges,
                           pce_degree=degree,
                           model_degrees=model_degrees)


if __name__ == "__main__":
    pass
