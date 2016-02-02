import numpy
import matplotlib.pyplot as plt

image = numpy.random.uniform(size=(149, 24))

plt.figure()
fig = plt.imshow(image, cmap=plt.cm.hot_r, interpolation='none')
plt.colorbar(fig, orientation='vertical')
plt.show()

fig = plt.imshow(image, cmap=plt.cm.hot_r, interpolation='nearest', aspect='auto')
plt.xticks(numpy.arange(0, 24, 1))
plt.colorbar(fig, orientation='vertical')
plt.show()