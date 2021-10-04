Plots
=====

The provided plotting methods are best used in conjunction with the rest of the UQ components.

How to Use
----------
Each plotting method takes a matplotlib axes object as its first argument. The specified plot will be drawn to that
object. One easy way of getting an axes object is to use the ``matplotlib.pyplot.subplots()`` method. This will return
a figure object and an axes object (or a list of objects).



.. code:: python

   fig, ax = matplotlib.pyplot.subplots()

   plots.contour_plot(ax, X, Y)




Plots API
---------
.. automodule:: uq_methods.plots
   :members: