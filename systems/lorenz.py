import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

def system(z, t, sigma, beta, rho):
	u, v, w = z
	du = -sigma*(u-v)
	dv = rho*u - v - u*w
	dw = -beta*w + u*v
	return du, dv, dw

def dataset(tmax: int, n: int):
	sigma, beta, rho = 10, 2.667, 28
	u0, v0, w0 = 0, 1, 1.05
	t = np.linspace(0, tmax, n)
	f = odeint(system, (u0, v0, w0), t, args=(sigma, beta, rho))
	X, Y = f[:-1].T, f[1:].T
	return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

if __name__ == '__main__':
	tmax, n = 100, 10000
	X, _ = dataset(tmax, n)
	x, y, z = X[0].numpy(), X[1].numpy(), X[2].numpy()
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	s = 10
	c = np.linspace(0, 1, n)
	for i in range(0,n-s,s):
		ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=(1,c[i],0), alpha=0.4)

	ax.set_axis_off()
	plt.show()
