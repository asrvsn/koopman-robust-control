'''
2D Burgers Equation, taken from https://github.com/FlorinGh/12-steps-to-navier-stokes/blob/master/stp8-burgers-equation-2D/2D_Burger_Equation.py
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nx = 51
ny = 51
nt = 1200

dx = 2.0/(nx-1)
dy = 2.0/(ny-1)
sigma = 0.0009
nu = 0.005
dt = sigma*dx*dy/nu

x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)

def step(un: np.ndarray, vn: np.ndarray):
	u = un.copy()
	v = vn.copy()
	
	u[1:-1,1:-1] = un[1:-1,1:-1] - \
	dt/dx*un[1:-1,1:-1]*(un[1:-1,1:-1]-un[0:-2,1:-1]) - \
	dt/dy*vn[1:-1,1:-1]*(un[1:-1,1:-1]-un[1:-1,0:-2]) + \
	nu*dt/dx**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]) + \
	nu*dt/dy**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])
	
	v[1:-1,1:-1] = vn[1:-1,1:-1] - \
	dt/dx*un[1:-1,1:-1]*(vn[1:-1,1:-1]-vn[0:-2,1:-1]) - \
	dt/dy*vn[1:-1,1:-1]*(vn[1:-1,1:-1]-vn[1:-1,0:-2]) + \
	nu*dt/dx**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1]) + \
	nu*dt/dy**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])
	
	u[0,:] = 1
	u[-1,:] = 1
	u[:,0] = 1
	u[:,-1] = 1
		
	v[0,:] = 1
	v[-1,:] = 1
	v[:,0] = 1
	v[:,-1] = 1

	return u, v

def dataset(nt=1200):

	# Initialization
	u = np.ones((ny,nx))
	v = np.ones((ny,nx))
	out = np.full((nt, u.shape[0], u.shape[1]), np.nan)

	# Assign initial conditions
	u[int(0.5/dy):int(1/dy+1), int(0.5/dx):int(1/dx+1)] = 2
	v[int(0.5/dy):int(1/dy+1), int(0.5/dx):int(1/dx+1)] = 2
	out[0] = u

	# Solve
	for n in range(1, nt):
		u, v = step(u, v)
		out[n] = u

	return out

if __name__ == '__main__':
	# Simulate 
	nt = 1200
	data = dataset(nt=nt)

	# Plot the initial condition
	fig  = plt.figure(figsize = (11,7), dpi = 100)
	ax = Axes3D(fig)
	X, Y = np.meshgrid(x,y)
	ax.plot_wireframe(X, Y, data[0])
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Burgers Equation: Initial condition')
	ax.set_zlim(1,2.0)

	# Plot the results
	fig = plt.figure(figsize = (11,7), dpi = 100)
	ax = Axes3D(fig)
	X, Y = np.meshgrid(x,y)
	ax.plot_wireframe(X,Y,data[-1])
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title(f'Burgers Equation: Solution after {nt} steps')
	ax.set_zlim(1,2.0)

	plt.show()