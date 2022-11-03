from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from sklearn.model_selection import train_test_split

import hill_data
from trata import composite_samples

DO_PLOTS = True

for i, (X_ob, Y_ob) in enumerate(zip(hill_data.X_obs, hill_data.Y_obs)):
    print(i)
    results = composite_samples.parse_file('{}_results.csv'.format(i), 'csv')
    points = results.get_points(['a', 'b', 'c'])
    Y = results.get_points(['output_0'])

    X_train, X_test, Y_train, Y_test = train_test_split(points, Y)
    model = hill_data.MODEL_TYPE(**hill_data.MODEL_ARGS)

    model.fit(X_train, Y_train)
    with open("{}_{}.mdl".format(i, hill_data.MODEL_TYPE), 'wb') as fh:
        pickle.dump(model, fh)
