"""
This script draws a plot showing a colorbar with custom limits.
Question: http://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib
		  http://stackoverflow.com/questions/21952100/setting-the-limits-on-a-colorbar-in-matplotlib
		  http://stackoverflow.com/questions/5826592/python-matplotlib-colorbar-range-and-display-values
"""

import matplotlib.pyplot as plt
import numpy

def plot_heat_map(v_min, v_max, data):
    plt.figure()
    fig = plt.imshow(data, vmin=v_min, vmax=v_max)
    plt.colorbar(fig)
    plt.show()
    plt.close()
	
# Data generated as show in http://matplotlib.org/examples/pylab_examples/contourf_demo.html

delta = 0.025

x = y = numpy.arange(-3.0, 3.01, delta)
X, Y = numpy.meshgrid(x, y)
Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z1 - Z2)

min_value = numpy.min(Z)
max_value = numpy.max(Z)

print min_value, max_value
plot_heat_map(min_value/2, max_value/2, Z)