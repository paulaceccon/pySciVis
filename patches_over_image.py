"""
This script simply customize a grid with custom major and minor ticks.
It also adds a margin to the image and uses the 'ggplot' style.
"""

import numpy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches

# Sample data
x = y = numpy.arange(-3.0, 3.0, 0.025)
X, Y  = numpy.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z2 - Z1)

# Choose random points based on the image size
x_steps, y_steps = Z.shape
n = x_steps * y_steps * 0.20
idx = numpy.arange(x_steps * y_steps).reshape(x_steps, y_steps)
idx_x, idx_y = numpy.unravel_index(numpy.random.choice(x_steps * y_steps, n, replace=False), idx.shape)
idx_x = numpy.concatenate((idx_x, numpy.array([x_steps/2])), axis=0)
idx_y = numpy.concatenate((idx_y, numpy.array([y_steps/2])), axis=0)

plt.figure()
fig = plt.imshow(Z, origin='lower', extent=[-3, 3, -3, 3], vmax=abs(Z).max(), vmin=-abs(Z).max())
plt.axes().set_aspect('equal')                                           

ax = plt.gca()

idx = [idx_x, idx_y]
for p in range(len(idx[0])):
    i = idx[0][p]
    j = idx[1][p] 
    ax.add_patch(patches.Rectangle((x[i], y[j]), 0.025, 0.025, facecolor='k', zorder=1))                                                    

plt.show()
plt.close()
