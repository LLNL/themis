from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.gaussian_process import GaussianProcessRegressor

from trata import composite_samples, adaptive_sampler
from themis import Run, user_utils

NUM_TOTAL = 5


def prep_ensemble():
    pass


def prep_run():
    pass


def post_run():
    z = 1e99
    try:
        with open('coderun.success', 'r') as f:
            line = f.readline()
            z = line.split()[0]
    except IOError:
        print('Error reading "coderun.success"')
    return z


def post_ensemble():
    mgr = user_utils.themis_handle()
    df = mgr.as_dataframe()
    if len(df.index) < NUM_TOTAL:
        X = df[["x", "y"]].to_numpy()
        Y = df[["result"]].to_numpy()
        model = GaussianProcessRegressor()
        model.fit(X, Y)

        Samples = composite_samples.Samples()
        Samples.set_continuous_variable('x', -2, 0, 2)
        Samples.set_continuous_variable('y', -2, 0, 2)

        Samples.generate_samples(['x', 'y'], adaptive_sampler.ActiveLearningSampler(),
                                 model=model, X=X, Y=Y, num_points=1, box=[[-2, 2], [-2, 2]], num_cand_points=100)

        runs = [Run(sample, 'rosenbrock_input_deck') for sample in Samples]

        mgr.add_runs(runs)
        print("ADDED POINTS {}".format(runs))
    else:
        print("*** ADAPTIVE SAMPLING COMPLETE ***")
