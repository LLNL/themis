"""
This package contains LLNL's UQ Pipeline (UQP).

This package is composed of four subpackages: sampling,
sensitivity, surrogate_model, and uq_component. These subpackages
are often referred to as "components".

The sampling component is used to generate sample points in order to explore a parameter space.
For instance, if a simulation takes two inputs, x and y, and you want to run a set of simulations
with x-values between 5 and 20 and y-values between 0.1 and 1000, the sampling component
can generate sample points (which in this case means (x,y) pairs) for you.
You can specify how many total sample points you want, and how you want them to be chosen;
the sampling component offers a large number of different sampling strategies.
If, on the other hand, you already have sample points you wish to use,
the component can simply read them in from a file.

The surrogate_model subpackage is used to generate surrogate models as part of the ensemble generation process
and can be utilzed as part of sensitivity analysis and uncertainty quantification analysis.
"""

__all__ = ["sampling", "sensitivity", "surrogate_model", "uq_component"]

__version__ = 1.0

__authors__ = [
    "David Domyancic",
    "Andrew Fillmore",
    "Keith Healy",
    "Scott Brandon",
    "Paul Minner",
    "James Corbett"
]
