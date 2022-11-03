from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hill_data
import themis
from trata import composite_samples, sampler

for i, (X_ob, Y_ob) in enumerate(zip(hill_data.X_obs, hill_data.Y_obs)):
    print(i)
    Samples = composite_samples.Samples()
    Samples.set_continuous_variable('a', 2, 2, 10)
    Samples.set_continuous_variable('b', 0, 0, 3)
    Samples.set_continuous_variable('c', 1, 1, 20)
    Samples.set_continuous_variable('x', 0, X_ob, 3)

    Samples.generate_samples(['a', 'b', 'c'], sampler.LatinHyperCubeSampler(), num_points=hill_data.N)
    Samples.generate_samples(['x'], sampler.DefaultValueSampler(), num_points=hill_data.N)

    runs = [themis.Run(sample, args='--input_deck input_deck') for sample in Samples]

    ensemble = themis.Themis.create_overwrite(
        application='hill_function.exe',
        runs=runs,
        run_copy='hill_function.exe',
        run_dir_names='ens_' + str(i) + '/{run_id}',
        app_interface='hill_interface.py',
        run_parse='input_deck',
        app_is_batch_script=False,
        setup_dir='.themis_setup_{}'.format(i),
        max_restarts=10
    )

    # ensemble.dry_run()
    ensemble.execute_local(blocking=True)

    with open('{}_results.csv'.format(i), 'w') as stream:
        ensemble.write_csv(stream)
