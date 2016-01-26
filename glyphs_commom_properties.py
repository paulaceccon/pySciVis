"""
This script draws a plot composed by glyphs.
The properties are common for the glyphs.
Question: http://stackoverflow.com/questions/29866592/draw-a-plot-of-glyphs-in-matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge, Circle
from math import degrees, pi

fig, ax = plt.subplots()
wedges = []
circles = []
colors = []

for x in numpy.arange(0, 3.3, .3):
    for y in numpy.arange(0, 3.3, .3):
		theta, phi = numpy.random.random(2) # Deviation and mean angle
		opacity = theta 					# [0.0, 1.0]
		theta *= numpy.pi / 2				# [0, pi/2]
		phi *= numpy.pi / 2					# [0, pi/2]
		for v in (0, pi):
			wedges.append(Wedge((x, y),
					      .15,
                          degrees(v + phi - theta),
                          degrees(v + phi + theta)))
			circles.append(Circle((x, y),
                           .15))
			colors.append(0.5)               

collection = PatchCollection(circles, cmap = plt.cm.hot_r, alpha=0.2)
collection.set_array(numpy.array(colors))
collection.set_clim(0.0, numpy.pi/2)
collection.set_edgecolor('none')
ax.add_collection(collection)

collection = PatchCollection(wedges, cmap = plt.cm.hot_r, alpha=0.8)
collection.set_array(numpy.array(colors))
collection.set_clim(0.0, numpy.pi/2)
collection.set_edgecolor('none')
ax.add_collection(collection)

plt.colorbar(collection, orientation='horizontal')

ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_aspect('equal')
plt.show()