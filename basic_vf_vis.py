"""
This script presents simple functions to manipulate and display vector fields.
"""

import numpy as np
import pylab as plt
import matplotlib.colors as colors

from scipy.interpolate import RegularGridInterpolator

plt.style.use('ggplot')

def regrid(data, out_y, out_x):
	"""Given some data with dimensions given by data.shape,
    regrid this data to shape out_y, out_x.
    """

	m = max(data.shape[0], data.shape[1])
	y = np.linspace(0, 1.0/m, data.shape[0])
	x = np.linspace(0, 1.0/m, data.shape[1])
	interpolating_function = RegularGridInterpolator((y, x), data)
	
	xv, yv = np.meshgrid(np.linspace(0, 1.0/m, out_x), np.linspace(0, 1.0/m, out_y))
	
	return interpolating_function((yv, xv))


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
    rx = 82
    ry = 72
    
    vx = np.loadtxt('Data/vf_x.csv', delimiter=',')
    vy = np.loadtxt('Data/vf_y.csv', delimiter=',')
    
    plot_quiver(vx, vy, 'Original') 
    
    vx = regrid(vx, ry, rx)
    vy = regrid(vy, ry, rx)
            
        
    plot_quiver(vx, vy, 'Quiver')  
    plot_streamlines(vx, vy, 'Streamplot')	
    plot_contour(vx, vy, 'Contour')