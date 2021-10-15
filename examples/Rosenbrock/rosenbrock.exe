#!/usr/bin/env python

import sys

import numpy as np

with open(sys.argv[1], 'r') as f:
    _, value = f.readline().split('=')
    x = float(value)
    _, value = f.readline().split('=')
    y = float(value)

z = np.power((1 - x), 2.0) + (100 * np.power(y - np.power(x, 2.0), 2.0))

with open('coderun.success', 'w') as f:
    f.write('  {} f1\n'.format(z))
