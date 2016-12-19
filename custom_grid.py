"""
This script simply customize a grid with custom major and minor ticks.
It also adds a margin to the image and uses the 'ggplot' style.
"""

import numpy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.style.use('ggplot')

# Sample data
x = y = numpy.arange(-3.0, 3.0, 0.025)
X, Y  = numpy.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2 - Z1

plt.figure()
fig = plt.imshow(Z, origin='lower', extent=[-3, 3, -3, 3], vmax=abs(Z).max(), vmin=-abs(Z).max())

# Add some margin
l, r, b, t = plt.axis()
dx, dy = r-l, t-b
plt.axis([l-0.1*dx, r+0.1*dx, b-0.1*dy, t+0.1*dy])

plt.axes().set_aspect('equal')
    
major_ticks = numpy.arange(-3.0, 3.0, 5)                                              
minor_ticks = numpy.arange(-3.0, 3.0, 0.025)                                               

ax = plt.gca()
ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)                                           
ax.set_yticks(major_ticks)                                                       
ax.set_yticks(minor_ticks, minor=True)                                                                                      

ax.grid(which='both')    
ax.grid(which='minor', alpha=0.3)                                                
ax.grid(which='major', alpha=0.5)

for axi in (ax.xaxis, ax.yaxis):
    for tic in axi.get_major_ticks():
        tic.tick1On  = tic.tick2On  = False
        tic.label1On = tic.label2On = False
    for tic in axi.get_minor_ticks():
        tic.tick1On  = tic.tick2On  = False
        tic.label1On = tic.label2On = False


plt.colorbar(fig)
plt.show()
plt.close()
