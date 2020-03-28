'''
Algorithms for computing dynamical systems operators.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from tqdm import tqdm

from sampler.features import *
from sampler.utils import *

def dmd(X: torch.Tensor, Y: torch.Tensor, operator='K'):
	X, Y = X.detach(), Y.detach() 
	C_XX = X@X.t()
	C_XY = X@Y.t()
	if operator == 'P': C_YX = C_XY.t()
	P = torch.pinverse(C_XX)@C_XY
	return P.t()

def dmdc(sys: Callable, ics: list, u: torch.Tensor, obs: Observable):
	'''
	u: N (simulations) x T (control inputs)
	'''
	V, W = [], []
	for i in range(len(ics)):
		for j in range(u.shape[0]):
			Xi, Yi = sys(ics[i], u[j,:])
			u_n = u[j,:Xi.shape[1]].unsqueeze(0)
			Vi = torch.cat((obs(Xi), u_n), axis=0)
			Wi = torch.cat((obs(Yi), Xi), axis=0)
			V.append(Vi)
			W.append(Wi)
	V = torch.cat(tuple(V), axis=1)
	W = torch.cat(tuple(W), axis=1)
	ABC = W@V.t()@torch.pinverse(V@V.t())
	A, B = ABC[:obs.k, :obs.k], ABC[:obs.k, obs.k:]
	return A, B

def kdmd(X: torch.Tensor, Y: torch.Tensor, k: Kernel, epsilon=0, operator='K'):
	X, Y = X.detach(), Y.detach() 
	n, device = X.shape[0], X.device
	G_XX = k.gramian(X, X)
	G_XY = k.gramian(X, Y)
	if operator == 'K':
		G_XY = G_XY.t()
	P = torch.mm(torch.pinverse(G_XX + epsilon*torch.eye(n, device=device)), G_XY)
	return P

def worker(x0, y0, P, obs, t):
	init = torch.Tensor([[x0], [y0]])
	Z = obs.extrapolate(P, init, t)
	return Z.numpy()

def sample_2d_dynamics(
		P: torch.Tensor, obs: Observable, t: int,
		x_range: tuple, y_range: tuple, n_x: int, n_y: int, 
	):
	'''
	Sample trajectories from Koopman operator for 2d system.
	'''
	assert x_range[0] < x_range[1]
	assert y_range[0] < y_range[1]
	trajectories = []

	with tqdm(total=n_x*n_y, desc='Trajectories') as pbar:
		def finish(result):
			trajectories.append(result)
			pbar.update(1)

		# https://github.com/pytorch/pytorch/issues/973
		# torch.multiprocessing.set_sharing_strategy('file_system')
		with multiprocessing.Pool() as pool:
			for x0 in np.linspace(x_range[0], x_range[1], n_x):
				for y0 in np.linspace(y_range[0], y_range[1], n_y):
					pool.apply_async(worker, args=(x0, y0, P, obs, t), callback=finish)
			pool.close()
			pool.join()

	return trajectories


if __name__ == '__main__':
	import systems.vdp as vdp
	import systems.duffing as duffing

	set_seed(9001)

	def dmd_test(name, obs, X, Y):
		PsiX, PsiY = obs(X), obs(Y)
		P = dmd(PsiX, PsiY)
		print('Spectral radius:', spectral_radius(P))
		assert P.shape[0] == P.shape[1]

		plt.figure(figsize=(8,8))
		plt.title(f'{name}: Original series')
		plt.plot(Y[0], Y[1])
		plt.figure(figsize=(8,8))
		plt.title(f'{name}: eDMD prediction')
		Yp = obs.preimage(torch.mm(P, PsiX))
		plt.plot(Yp[0], Yp[1])

		plt.figure(figsize=(8,8))
		plt.title(f'{name}: eDMD extrapolation')
		Yp = obs.extrapolate(P, X, X.shape[1])
		plt.plot(Yp[0], Yp[1])

	# print('VDP DMD test')
	# mu = 2.0
	# X, Y = vdp.dataset(mu, n=10000)
	# p, d, k = 4, X.shape[0], 5
	# obs = PolynomialObservable(p, d, k)
	# dmd_test('VDP', obs, X, Y)

	# print('Duffing DMD test')
	# X, Y = duffing.dataset(80, 4000, gamma=0.0, x0=2.0, xdot0=2.0)
	# # p, d, tau = 11, X.shape[0], 0
	# # obs1 = PolynomialObservable(p, d, 62)
	# # obs2 = DelayObservable(obs1.k, tau)
	# # obs = ComposedObservable([obs1, obs2])
	# p, d, k = 5, X.shape[0], 15
	# obs = PolynomialObservable(p, d, k)
	# dmd_test('Duffing', obs, X, Y)

	# print('kDMD test')
	# mu = 1
	# X, Y = vdp.dataset(mu)
	# sigma = np.sqrt(0.3)
	# k = GaussianKernel(sigma)
	# P = kdmd(X, Y, k)
	# assert P.shape[0] == P.shape[1] == X.shape[0]


	plt.show()
