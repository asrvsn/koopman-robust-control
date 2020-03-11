'''
Algorithms for computing dynamical systems operators.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

from features import *
from utils import set_seed

def dmd(X: torch.Tensor, Y: torch.Tensor, operator='K', spectral_constraint=None):
	X, Y = X.detach(), Y.detach() 
	C_XX = torch.mm(X, X.t())
	C_XY = torch.mm(X, Y.t())
	if operator == 'P': C_YX = C_XY.t()
	P = torch.mm(torch.pinverse(C_XX), C_XY)
	return P.t()

def kdmd(X: torch.Tensor, Y: torch.Tensor, k: Kernel, epsilon=0, operator='K'):
	X, Y = X.detach(), Y.detach() 
	n, device = X.shape[0], X.device
	G_XX = k.gramian(X, X)
	G_XY = k.gramian(X, Y)
	if operator == 'K':
		G_XY = G_XY.t()
	P = torch.mm(torch.pinverse(G_XX + epsilon*torch.eye(n, device=device)), G_XY)
	return P

def is_semistable(P: torch.Tensor, eps=1e-3):
	return torch.norm(P, p=2).item() <= 1.0 + eps

if __name__ == '__main__':
	import systems.vdp as vdp
	import systems.duffing as duffing

	set_seed(9001)

	def dmd_test(name, obs, X, Y):
		PsiX, PsiY = obs(X), obs(Y)
		P = dmd(PsiX, PsiY)
		print('Spectral norm:', torch.norm(P, p=2).item())
		assert P.shape[0] == P.shape[1]

		plt.figure(figsize=(8,8))
		plt.title(f'{name}: Original series')
		plt.plot(Y[0], Y[1])
		plt.figure(figsize=(8,8))
		plt.title(f'{name}: eDMD prediction')
		Yp = obs.preimage(torch.mm(P, PsiX))
		# print('prediction loss: ', (Yp - Y).norm().item())
		plt.plot(Yp[0], Yp[1])
		plt.figure(figsize=(8,8))
		plt.title(f'{name}: eDMD extrapolation')
		Yp = obs.extrapolate(P, X, X.shape[1])
		# print('extrapolation loss: ', (Yp - Y).norm().item())
		plt.plot(Yp[0], Yp[1])

	print('VDP DMD test')
	mu = 2.0
	X, Y = vdp.dataset(mu, n=10000)
	p, d, k = 4, X.shape[0], 5
	obs = PolynomialObservable(p, d, k)
	dmd_test('VDP', obs, X, Y)

	print('Duffing DMD test')
	X, Y = duffing.dataset(400, 10000, gamma=0.37)
	# p, d, tau = 11, X.shape[0], 0
	# obs1 = PolynomialObservable(p, d, 62)
	# obs2 = DelayObservable(obs1.k, tau)
	# obs = ComposedObservable([obs1, obs2])
	p, d, k = 6, X.shape[0], 27
	obs = PolynomialObservable(p, d, k)
	dmd_test('Duffing', obs, X, Y)

	print('kDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	sigma = np.sqrt(0.3)
	k = GaussianKernel(sigma)
	P = kdmd(X, Y, k)
	assert P.shape[0] == P.shape[1] == X.shape[0]

	plt.show()
