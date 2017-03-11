"""
This script presents simple functions to manipulate and display vector fields.
"""

import numpy as np
import pylab as plt
import matplotlib.colors as colors

from scipy.interpolate import RegularGridInterpolator

plt.style.use('ggplot')


def plot_quiver(vx, vy, file_name):
	"""Given a vector field specified by vx, vy, display it using
	the quiver function of matplotlib.
	It also specifies the correct axis limits to obtain a better plot aspect.
	Moreover, it shows how to normalize the data among specific ranges.
	"""
	x, y = np.mgrid[0:vx.shape[0]:vx.shape[0]*1j, 0:vx.shape[1]:vx.shape[1]*1j]
	mag = np.hypot(vx, vy)

	plt.figure()
	plt.quiver(x, y, vx, vy, mag, clim=[0, 70])
	plt.colorbar(orientation='vertical')
	plt.xlim(0, vx.shape[0])
	plt.ylim(0, vx.shape[1])
	plt.xticks([])
	plt.yticks([])
	plt.axes().set_aspect('equal')
	plt.savefig(file_name+'.png', bbox_inches='tight', dpi=400)	
    
    
def plot_streamlines(vx, vy, file_name):
	"""Given a vector field specified by vx, vy, display it using
	the streamline function of matplotlib.
	It also specifies the correct axis limits to obtain a better plot aspect.
	Moreover, it shows how to normalize the data among specific ranges.
    """
	xrange = np.linspace(0, vx.shape[0], vx.shape[0]);
	yrange = np.linspace(0, vx.shape[1], vx.shape[1]);

	x, y = np.meshgrid(xrange, yrange)
	mag = np.hypot(vx, vy)
	scale = 2
	lw = scale * mag / mag.max()
	
	plt.figure()
	plt.streamplot(x, y, vx.T, vy.T, color=np.hypot(vx, vy).T, density=3, linewidth=lw.T, arrowsize=scale, norm=plt.Normalize(0, 70))
	plt.colorbar(orientation='vertical')
	plt.xlim(0, vx.shape[0])
	plt.ylim(0, vx.shape[1])
	plt.xticks([])
	plt.yticks([])
	plt.axes().set_aspect('equal')
	plt.savefig(file_name+'.png', bbox_inches='tight', dpi=400)	


def plot_contour(vx, vy, file_name):
	"""Given a vector field specified by vx, vy, display it using
	the streamline function of matplotlib.
	It also specifies the correct axis limits to obtain a better plot aspect.
	Moreover, it shows how to normalize the data among specific ranges.
	"""
	x, y = np.mgrid[0:vx.shape[0]:vx.shape[0]*1j, 0:vx.shape[1]:vx.shape[1]*1j]
	mag = np.hypot(vx, vy)

	plt.figure()
	plt.contourf(x, y, mag, 12, vmin=0, vmax=70)
	plt.colorbar(orientation='vertical')
	plt.xticks([])
	plt.yticks([])
	plt.axes().set_aspect('equal')
	plt.savefig(file_name+'.png', bbox_inches='tight', dpi=400)	
    


if __name__ == '__main__': 
    vx = np.loadtxt('Data/vf_x.csv', delimiter=',')
    vy = np.loadtxt('Data/vf_y.csv', delimiter=',')
        
    plot_quiver(vx, vy, 'Quiver')  
    plot_streamlines(vx, vy, 'Streamplot')	
    plot_contour(vx, vy, 'Contour')