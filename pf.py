'''
Algorithms for computing Perron-Frobenius operator.

PyTorch adaptation of [d3s](https://github.com/sklus/d3s), Klus et al
'''
import torch
import numpy as np

from features import *

def edmd(X: torch.Tensor, Y: torch.Tensor, psi: Observable):
	PsiX = psi(X)
	PsiY = psi(Y)
	C_XX = torch.mm(PsiX, PsiX.t())
	C_YX = torch.mm(PsiX, PsiY.t()).t()
	P = torch.mm(torch.pinverse(C_XX), C_YX)
	Phi = torch.mm(Y, torch.pinverse(PsiY)).t() # preimage map
	return P, Phi

def kdmd(X: torch.Tensor, Y: torch.Tensor, k: Kernel, epsilon=0):
	n, device = X.shape[0], X.device
	G_XX = gramian(X, X, k)
	G_XY = gramian(X, Y, k).t()
	P = torch.mm(torch.pinverse(G_XX + epsilon*torch.eye(n, device=device)), G_XY)
	return P

if __name__ == '__main__':
	import systems.vdp as vdp

	print('eDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	obs = PolynomialObservable(3, X.shape[1])
	P, _ = edmd(torch.from_numpy(X), torch.from_numpy(Y), obs)
	assert P.shape[0] == P.shape[1] == X.shape[0]

	print('kDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	sigma = np.sqrt(0.3)
	k = GaussianKernel(sigma)
	P = kdmd(torch.from_numpy(X), torch.from_numpy(Y), k)
	assert P.shape[0] == P.shape[1] == X.shape[0]

	# TODO: prediction tests
