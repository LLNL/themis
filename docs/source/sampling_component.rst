.. _sampling_component:

Sampling
========

The sampling component is used to generate sample points in order to explore a parameter space. For instance, if a simulation takes two inputs, x and y, and you want to run a set of simulations with x-values between 5 and 20 and y-values between 0.1 and 1000, the sampling component can generate sample points (which in this case means (x,y) pairs) for you. You can specify how many total sample points you want, and how you want them to be chosen--the sampling component offers a large number of different sampling strategies. If, on the other hand, you already have sample points you wish to use, the component can simply read them in from a file. 

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   sampling_tutorial
   sampling_sampler
   sampling_adapt
   sampling_composite
