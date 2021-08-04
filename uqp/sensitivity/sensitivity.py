from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Tkagg')

import copy

import matplotlib.pyplot as plt
import networkx
import numpy as np
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import uqp.sampling.sampler
import uqp.surrogate_model.pce_model


def _var_multiplier(plist):
    multiplier = 1.0
    for p in plist:
        if p == 0:
            pass
        elif p == 1:
            multiplier *= 1. / 3.
        elif p == 2:
            multiplier *= 1. / 5.
        elif p == 3:
            multiplier *= 1. / 7.
        elif p == 4:
            multiplier *= 1. / 9.
        elif p == 5:
            multiplier *= 1. / 11.
        elif p == 6:
            multiplier *= 1. / 13.
        elif p == 7:
            multiplier *= 1. / 15.
        elif p == 8:
            multiplier *= 1. / 17.
        else:
            raise NotImplementedError('Multiplier only exists for 0..8')

    return multiplier


def _make_dt_graph(names, coefficients, interactions, num_degree):
    dt_graph = {degree: {} for degree in range(1, num_degree + 1)}
    for name in names:
        dt_graph[1][(name,)] = 0.0

    for coefficient, interaction in zip(coefficients, interactions):
        mult = _var_multiplier(interaction.values())
        var_term = mult * coefficient ** 2
        x_key = [names[key] for key, value in filter(lambda x: x[1] != 0, interaction.items())]
        x_key = tuple(sorted(x_key))
        num_terms = len(x_key)
        if x_key in dt_graph[num_terms]:
            dt_graph[num_terms][x_key] += var_term
        else:
            dt_graph[num_terms][x_key] = var_term

    total_var = 0.0

    for degree in dt_graph.keys():
        total_var += sum(dt_graph[degree].values())

    for degree in dt_graph.keys():
        for p_key in dt_graph[degree].keys():
            dt_graph[degree][p_key] = dt_graph[degree][p_key] / total_var

    return dt_graph


def variance_network_plot(ax, surrogate_model, degree=2, max_node_size=4000.0, max_edge_size=100.0, alpha=.5,
                          label_size=12, draw_node_labels=False, draw_contrib_labels=False):
    for name, axes in zip(surrogate_model.Y_names, ax):

        model = surrogate_model._models[name]

        if not isinstance(model, uqp.surrogate_model.pce_model.PolynomialChaosExpansionModel):
            raise ValueError('Can only create Variance Network Plot on Polynomial Chaos Expansion surrogate model. '
                             'Was given type {}'.format(type(model)))
        axes.set_title(name)

        coefficients = model.coefficients
        interactions = model.interactions
        num_degree = model.num_degrees

        degrees_to_plot = range(1, num_degree + 1) if degree is None else [1, degree]

        for deg in degrees_to_plot:
            if deg > num_degree:
                raise ValueError("Cannot create plot. Given degree is higher that degree of surrogate model.\n"
                                 "Given 'degree'={}\n"
                                 "Expected 'degree'<={}".format(degree, num_degree))

        names = list(surrogate_model.X_names)

        dt_graph = _make_dt_graph(names, coefficients, interactions, num_degree)

        # Make Network
        num_params = len(names)

        m = num_params * (num_params - 1)
        graph = networkx.gnm_random_graph(num_params, m)
        pos = networkx.circular_layout(graph, scale=1.0)
        pos2 = networkx.circular_layout(graph, center=(0, .075))

        np_node_sizes = np.array([dt_graph[1][(name,)] for name in names])
        node_size_factor = max_node_size / np_node_sizes.max()
        np_node_sizes = node_size_factor * np_node_sizes

        networkx.draw_networkx_nodes(graph, pos, node_size=np_node_sizes,
                                     node_shape='o',
                                     node_color='salmon',
                                     linewidths=0.0,
                                     alpha=alpha, ax=axes)
        if draw_node_labels:
            networkx.draw_networkx_labels(graph, pos2, {i: name for i, name in enumerate(names)},
                                          font_size=label_size, ax=axes)

        ##################################################
        colors = ['black', 'red', 'blue', 'green']
        ##################################################

        dt_all_edge_labels = {}

        leg_artists = []
        leg_labels = []

        for deg in degrees_to_plot:
            ls_edges = []
            ls_edge_widths = []
            ls_edge_contribs = []
            dt_edge_contribs = {}
            dt_edge_labels = {}
            for key in dt_graph[deg].keys():
                num_nodes = len(key)

                frac = dt_graph[deg][key]
                edge_width = frac

                edge_pair = tuple([names.index(key[0]), names.index(key[-1])])

                ls_edges.append(edge_pair)
                ls_edge_widths.append(edge_width)
                ls_edge_contribs.append(frac)

                if edge_pair in dt_edge_contribs:
                    dt_edge_contribs[edge_pair] += dt_graph[deg][key] / num_nodes
                else:
                    dt_edge_contribs[edge_pair] = dt_graph[deg][key] / num_nodes

                dt_edge_labels[edge_pair] = '{:.2f}'.format(edge_width * 100)

                if num_nodes > 2:
                    for i in range(1, num_nodes):
                        edge_pair = tuple([names.index(key[i - 1]), names.index(key[i])])

                        ls_edges.append(edge_pair)
                        ls_edge_widths.append(edge_width)

                        if edge_pair in dt_edge_contribs:
                            dt_edge_contribs[edge_pair] += dt_graph[deg][key]
                        else:
                            dt_edge_contribs[edge_pair] = dt_graph[deg][key]

                        dt_edge_labels[edge_pair] = '{:.2f}'.format(edge_width * 100)

                dt_all_edge_labels.update(dt_edge_labels)

            np_edge_sizes = np.array(ls_edge_widths)
            edge_size_factor = max_edge_size / np_edge_sizes.max()
            np_edge_sizes = np_edge_sizes * edge_size_factor

            networkx.draw_networkx_edges(graph, pos,
                                         edgelist=ls_edges, width=np_edge_sizes,
                                         alpha=alpha,
                                         edge_color=colors[deg % 4],
                                         zorder=deg, ax=axes)
            if draw_contrib_labels:
                networkx.draw_networkx_edge_labels(graph, pos, edge_labels=dt_all_edge_labels,
                                                   font_size=label_size, label_pos=.6, ax=axes)

            # if draw_legend:
            #     for val, width in zip(ls_edges, ls_edge_widths):
            #         color = colors[deg % 4]
            #         leg_artists.append(plt.Line2D((0, 1), (1, 1), color=color, alpha=alpha, linewidth=width))
            #         leg_labels.append('{}'.format(val))

        # if draw_legend:
        #     xExtension = 0.0
        #     yExtension = 1.0
        #
        #     axes.legend(leg_artists, leg_labels, ncol=2, numpoints=1, labelspacing=2,
        #                 handletextpad=1, fontsize=12, bbox_to_anchor=(xExtension, yExtension))


def rank_order_plot(ax, surrogate_model):
    num_params = len(surrogate_model.X_names)
    num_ops = len(surrogate_model.Y_names)

    rank_table = np.tile(None, (num_params, num_ops))

    for i, name in enumerate(surrogate_model.Y_names):

        model = surrogate_model._models[name]

        if not isinstance(model, uqp.surrogate_model.pce_model.PolynomialChaosExpansionModel):
            raise ValueError('Can only create Rank Order Plot on Polynomial Chaos Expansion surrogate model. '
                             'Was given type {}'.format(type(model)))

        coefficients = model.coefficients
        interactions = model.interactions
        num_degree = model.num_degrees

        names = list(surrogate_model.X_names)

        dt_graph = _make_dt_graph(names, coefficients, interactions, num_degree)

        dt_nodes = dict(dt_graph[1])
        for edge in dt_graph[2].keys():
            for node in edge:
                dt_nodes[(node,)] += dt_graph[2][edge] / 2.0

        dt_ranking = {list(dt_nodes.keys())[j][0]: k for k, j in enumerate(np.argsort(dt_nodes.values()))}

        for j, x_name in enumerate(surrogate_model.X_names[::-1]):
            rank_table[j, i] = dt_ranking[x_name]

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.pcolormesh(rank_table.astype(float), edgecolors='None', lw=0.0, cmap=cm.gist_ncar_r)

    ax.hlines(np.arange(1, num_params + 1), 0, num_ops)
    ax.vlines(np.arange(1, num_ops + 1), 0, num_params)

    cbar = plt.colorbar(im, cax=cax, orientation='vertical', values=np.arange(num_params))

    dt_labels = dict(zip(range(num_params), [''] * num_params))
    dt_labels[0] = 'Least\nSensitive'
    dt_labels[num_params - 1] = 'Most\nSensitive'

    cbar.ax.get_yaxis().set_ticks([])

    mat_values = [(num_params - im._A[i:i + num_ops]) for i in range(0, len(im._A), num_ops)]
    cols = [num_params for i in range(num_params)]
    rects = ax.barh(range(num_params), tuple(cols), 1.0, color='', align='center', linewidth=0)
    for r, rect in enumerate(rects):
        y_loc = rect.get_y() + 1.0
        for c in range(num_ops):
            x_loc = c + 0.5
            m_val = int(mat_values[r][c])
            ax.text(x_loc, y_loc, m_val, ha='center', va='center')

    for j in range(num_params):
        vertical_offset = (2 * j + 1) / (2. * num_params)

        cbar.ax.text(.5, vertical_offset, num_params - j, ha='center', va='center')
        cbar.ax.text(1.1, vertical_offset, dt_labels[j], ha='left', va='center', multialignment='center')

    ax.set_xticks(np.arange(num_ops) + 0.5)
    ax.set_xticklabels(surrogate_model.Y_names, rotation=35, ha='right')
    ax.set_xlabel('Outputs', ha='center')

    ax.set_yticks(np.arange(num_params) + 0.5)
    ax.set_yticklabels(surrogate_model.X_names[::-1])
    ax.set_ylabel('Parameters')

    ax.set_xlim([0, num_ops])
    ax.set_ylim([0, num_params])


def sensitivity_plot(ax, surrogate_model, num_plot_points, num_seed_points, seed=2018):
    X_range = surrogate_model.X_range

    np_seed_points = uqp.sampling.sampler.LatinHyperCubeSampler.sample_points(num_points=num_seed_points,
                                                                               box=X_range,
                                                                               seed=seed)

    COLORS = ['red', 'green', 'blue']

    for color_i, point in enumerate(np_seed_points):
        for i, (_, x_name) in enumerate(zip(point, surrogate_model.X_names)):
            sweep = np.linspace(X_range[i][0], X_range[i][1], num_plot_points)

            def gen():
                for el in sweep:
                    tmp_point = copy.deepcopy(point)
                    tmp_point[i] = el
                    yield tmp_point

            new_X = list(gen())
            for j, y_name in enumerate(surrogate_model.Y_names):
                ax[i][j].set_xlabel(x_name)
                ax[i][j].set_ylabel(y_name)
                ax[i][j].plot(sweep, surrogate_model[y_name].predict(new_X)[0],
                              color=COLORS[(color_i % len(COLORS)) - 1])


if __name__ == "__main__":
    import uqp.surrogate_model.surrogate_model
    import numpy as np

    X = np.random.rand(1000, 6)
    Y1 = np.prod([(np.sin(2 * np.pi * X[:, i]) - X[:, i + 1] ** 2.0) for i in range(5)], axis=0)
    Y2 = np.prod([(np.tanh(X[:, i]) - X[:, i + 1] ** .5) for i in range(5)], axis=0)
    Y = np.stack([Y1, Y2], axis=1)

    print('Fitting Model')
    model = uqp.surrogate_model.surrogate_model.SurrogateModel('pce',
                                                                [a for a in 'abcdef'.upper()],
                                                                ['Square', 'Square Root'],
                                                                X_range=[[0, 1]] * 6, num_degrees=3
                                                                ).fit(X, Y)
    fig, ax = plt.subplots(1, 3)

    fig.set_size_inches(20, 20)

    print('Plotting Variance Network')
    variance_network_plot(ax[:2], surrogate_model=model, alpha=.25, degree=2, draw_contrib_labels=True,
                          draw_node_labels=True)
    print('Plotting Rank Order')
    rank_order_plot(ax[2], surrogate_model=model)
    # print('Plotting Sensitivities')
    # sensitivity_plot(ax.reshape(6, 2), surrogate_model=model, num_plot_points=30, num_seed_points=12)
    print('Done')
    plt.show()
