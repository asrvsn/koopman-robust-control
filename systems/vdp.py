from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch

def system(mu: float):
	return lambda t, z: [z[1], mu*(1-z[0]**2)*z[1] - z[0]]

def dataset(mu: float, a=0, b=10):
	t = linspace(a, b, 500)
	sol = solve_ivp(system(mu), [a, b], [1, 0], t_eval=t)
	X, Y = sol.y[:, :-1], sol.y[:, 1:] # returns: d x n 
	return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

if __name__ == '__main__':
	mu = 3
	X, Y = dataset(mu)
	print(X.shape, Y.shape)
	plt.figure(figsize=(8,8))
	plt.plot(X[0], X[1])
	plt.figure(figsize=(8,8))
	plt.plot(Y[0], Y[1])
	plt.show()