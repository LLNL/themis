from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import themis
from trata import composite_samples, sampler

N = 3

Samples = composite_samples.Samples()
Samples.set_continuous_variable('x', -2, 0, 2)
Samples.set_continuous_variable('y', -2, 0, 2)

Samples.generate_samples(['x', 'y'], sampler.LatinHyperCubeSampler(), num_points=N)

runs = [themis.Run(sample, 'rosenbrock_input_deck') for sample in Samples]

mgr = themis.Themis.create_overwrite(
    os.path.join(os.path.dirname(__file__), 'rosenbrock.exe'),
    runs=runs,
    app_interface=os.path.join(os.path.dirname(__file__), 'rosenbrock_interface.py'),
    run_parse=os.path.join(os.path.dirname(__file__), 'rosenbrock_input_deck'),
    app_is_batch_script=False,
)

mgr.execute_local()
