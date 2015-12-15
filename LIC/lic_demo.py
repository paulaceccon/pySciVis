"""
This script comes from the SciPy Cookbook: 
--- http://scipy.github.io/old-wiki/pages/Cookbook/LineIntegralConvolution
It draws a vector field using Linear Integral Convolution.
To build the Cython file, use:
--- python setup.py build_ext --inplace
My improviment consists in tyring to get a better resolution when the vector field is 
not enough discretized, using some interpolation.
Question: http://scicomp.stackexchange.com/questions/14370/line-integral-convolution-lic-requirements
"""

import numpy as np
import pylab as plt

import lic_internal
import scipy.ndimage


def vector_flow(output_file, i_factor=1):
	# DPI resolution of the image to be saved
	dpi = 200

	file_path_x = "example/vector_field_x_0.csv"
	file_path_y = "example/vector_field_y_0.csv"

	vector_field_x = np.loadtxt(file_path_x, delimiter=",")
	vector_field_y = np.loadtxt(file_path_y, delimiter=",")

	x_steps, y_steps = vector_field_x.shape
	
	# Interpolation factor. For 1 no interpolation occurs.
	if i_factor > 1:
		vector_field_x = scipy.ndimage.zoom(vector_field_x, i_factor)
		vector_field_y = scipy.ndimage.zoom(vector_field_y, i_factor)

		x_steps *= i_factor
		y_steps *= i_factor


	# Putting data in the expected format
	vectors = np.zeros((x_steps, y_steps, 2), dtype=np.float32)

	vectors[...,0] += vector_field_y
	vectors[...,1] += vector_field_x
	
	texture = np.random.rand(x_steps,y_steps).astype(np.float32)

	kernellen=20
	kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
	kernel = kernel.astype(np.float32)

	image = lic_internal.line_integral_convolution(vectors, texture, kernel)
	mag = np.hypot(vector_field_x, vector_field_y)

	plt.jet()
	plt.figure()
	plt.axis('off')
	plt.imshow(texture, interpolation='nearest')
	plt.savefig(output_file+"-texture.png",dpi=dpi)

	
	plt.figure()
	fig = plt.quiver(vector_field_y, vector_field_x, mag)
	plt.colorbar()
	plt.savefig(output_file+".png",dpi=dpi)

	plt.bone()
	fig = plt.imshow(image, interpolation='nearest')
	# plt.colorbar()
	plt.savefig(output_file+"-flow.png",dpi=dpi)

if __name__ == '__main__':
	vector_flow("original")
	vector_flow("more_resolution", 4)