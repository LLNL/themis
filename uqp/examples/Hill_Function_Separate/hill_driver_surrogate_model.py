from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('/collab/usr/gapps/uq/')

from uqp.sampling import composite_samples
from uqp.surrogate_model import surrogate_model, surrogate_model_plots

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import hill_data

DO_PLOTS = True

for i, (X_ob, Y_ob) in enumerate(zip(hill_data.X_obs, hill_data.Y_obs)):
    print(i)
    results = composite_samples.parse_file('{}_results.csv'.format(i), 'csv')
    points = results.get_points(['a', 'b', 'c'])
    Y = results.get_points(['output_0'])

    X_train, X_test, Y_train, Y_test = train_test_split(points, Y)
    model = surrogate_model.SurrogateModel(hill_data.MODEL_TYPE, ['a', 'b', 'c'], ['output_0'],
                                           results.get_ranges(['a', 'b', 'c']),
                                           **hill_data.MODEL_ARGS)

    model.fit(X_train, Y_train)
    model.save('{}_{}.mdl'.format(i, hill_data.MODEL_TYPE))

    if DO_PLOTS:
        fig, ax = plt.subplots(1, 1)
        surrogate_model_plots.convergence_plot(model, X_test, Y_test, ax=ax)

        fig.savefig('{}_convergence_plot.png'.format(i))
        plt.close(fig)
