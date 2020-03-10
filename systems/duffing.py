import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def system(z, t, alpha, beta, gamma, delta, omega):
	x, y = z
	xdot = y
	ydot = -delta*y - alpha*x - beta*(x**3) + gamma*np.cos(omega*t)
	return xdot, ydot

def dataset(tmax: int, n: int, alpha=-1.0, beta=1.0, gamma=8.0, delta=0.3, omega=1.2):
	x0, y0 = 1, 0
	t = np.linspace(0, tmax, n)
	f = odeint(system, (x0, y0), t, args=(alpha, beta, gamma, delta, omega))
	X, Y = f[:-1].T, f[1:].T
	return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

if __name__ == '__main__': 
	t, n = 80, 4000

	X, Y = dataset(t, n, gamma=0.2)
	plt.figure(figsize=(8,8))
	plt.title('Period-1 oscillation')
	plt.plot(X[0], X[1])

	X, Y = dataset(t, n, gamma=0.28)
	plt.figure(figsize=(8,8))
	plt.title('Period-2 oscillation')
	plt.plot(X[0], X[1])

	X, Y = dataset(t, n, gamma=0.29)
	plt.figure(figsize=(8,8))
	plt.title('Period-4 oscillation')
	plt.plot(X[0], X[1])

	X, Y = dataset(t, n, gamma=0.37)
	plt.figure(figsize=(8,8))
	plt.title('Period-5 oscillation')
	plt.plot(X[0], X[1])

	X, Y = dataset(t, n, gamma=0.50)
	plt.figure(figsize=(8,8))
	plt.title('Chaos')
	plt.plot(X[0], X[1])

	X, Y = dataset(t, n, gamma=0.65)
	plt.figure(figsize=(8,8))
	plt.title('Period-2 oscillation (2)')
	plt.plot(X[0], X[1])

	plt.show()
