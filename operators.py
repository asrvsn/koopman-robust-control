'''
Algorithms for computing Perron-Frobenius operator.

PyTorch adaptation of [d3s](https://github.com/sklus/d3s), Klus et al
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

from features import *

def edmd(X: torch.Tensor, Y: torch.Tensor, psi: Observable, operator='K'):
	PsiX = psi(X)
	PsiY = psi(Y)
	C_XX = torch.mm(PsiX, PsiX.t())
	C_XY = torch.mm(PsiX, PsiY.t())
	if operator == 'P': C_YX = C_XY.t()
	P = torch.mm(torch.pinverse(C_XX), C_XY)
	return P

def kdmd(X: torch.Tensor, Y: torch.Tensor, k: Kernel, epsilon=0, operator='K'):
	n, device = X.shape[0], X.device
	G_XX = gramian(X, X, k)
	G_XY = gramian(X, Y, k)
	if operator == 'K':
		G_XY = G_XY.t()
	P = torch.mm(torch.pinverse(G_XX + epsilon*torch.eye(n, device=device)), G_XY)
	return P

if __name__ == '__main__':
	import systems.vdp as vdp

	print('eDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	p, d, k = 3, X.shape[0], 8
	obs = PolynomialObservable(p, d, k)
	P = edmd(X, Y, obs)
	assert P.shape[0] == P.shape[1] == k

	print('eDMD prediction')
	Z = obs.preimage(torch.mm(P, obs(X)))
	print('prediction loss: ', (Z - Y).norm().item())
	plt.figure(figsize=(8,8))
	plt.title('Original series')
	plt.plot(Y[0], Y[1])
	plt.figure(figsize=(8,8))
	plt.title('eDMD prediction')
	plt.plot(Z[0], Z[1])

	print('kDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	sigma = np.sqrt(0.3)
	k = GaussianKernel(sigma)
	P = kdmd(X, Y, k)
	assert P.shape[0] == P.shape[1] == X.shape[0]

	plt.show()
