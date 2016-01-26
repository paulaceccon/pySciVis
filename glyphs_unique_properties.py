"""
This script draws a plot composed by glyphs.
Each glyph has its own set of properties, such as alpha value, face color and edge color.
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
deviation = []

norm = mpl.colors.Normalize(0.0, numpy.pi/2)
cmap = plt.cm.hot_r
m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

for x in numpy.arange(0, 3.3, .3):
    for y in numpy.arange(0, 3.3, .3):
		theta, phi = numpy.random.random(2) # Deviation and mean angle
		opacity = theta 					# [0.0, 1.0]
		theta *= numpy.pi / 2				# [0, pi/2]
		phi *= numpy.pi / 2					# [0, pi/2]
		for v in (0, pi):
			color = m.to_rgba(phi)
			opacity = theta / (numpy.pi / 2)
			wedges.append(Wedge((x, y),
					      .15,
                          degrees(v + phi - theta),
                          degrees(v + phi + theta),
                          facecolor=color, alpha=opacity, edgecolor='none'))
			circles.append(Circle((x, y),
                           .15,
                           facecolor=color, alpha=0.2, edgecolor='none'))
		deviation.append(theta)               

m.set_array(deviation)

collection = PatchCollection(circles, match_original=True)
ax.add_collection(collection)

collection = PatchCollection(wedges, match_original=True)
ax.add_collection(collection)

plt.colorbar(m, orientation='horizontal')

ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_aspect('equal')
plt.show()