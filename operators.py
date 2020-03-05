'''
Algorithms for computing Perron-Frobenius operator.

PyTorch adaptation of [d3s](https://github.com/sklus/d3s), Klus et al
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

from features import *
from utils import set_seed

def dmd(X: torch.Tensor, Y: torch.Tensor, operator='K'):
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
	# Kernel only valid for operators with eigenvalues on unit circle.
	with torch.no_grad():
		(eig, _) = torch.eig(P)
		re, im = eig[:, 0], eig[:, 1]
		norm = torch.sqrt(re.pow(2) + im.pow(2))
		return (norm <= 1.0 + eps).all().item()

if __name__ == '__main__':
	import systems.vdp as vdp
	set_seed(9001)

	def dmd_test(obs, X, Y):
		PsiX, PsiY = obs(X), obs(Y)
		P = dmd(PsiX, PsiY)
		assert P.shape[0] == P.shape[1]

		plt.figure(figsize=(8,8))
		plt.title('Original series')
		plt.plot(Y[0], Y[1])
		plt.figure(figsize=(8,8))
		plt.title('eDMD prediction')
		Yp = obs.preimage(torch.mm(P, obs(X)))
		print('prediction loss: ', (Yp - Y).norm().item())
		plt.plot(Yp[0], Yp[1])
		plt.figure(figsize=(8,8))
		plt.title('eDMD extrapolation')
		Yp = obs.extrapolate(P, X, X.shape[1])
		print('extrapolation loss: ', (Yp - Y).norm().item())
		plt.plot(Yp[0], Yp[1])

	print('eDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	# p, d, k = 3, X.shape[0], 8
	# obs = PolynomialObservable(p, d, k)
	# dmd_test(obs, X, Y)
	obs = DelayObservable(3)
	dmd_test(obs, X, Y)

	print('kDMD test')
	mu = 1
	X, Y = vdp.dataset(mu)
	sigma = np.sqrt(0.3)
	k = GaussianKernel(sigma)
	P = kdmd(X, Y, k)
	assert P.shape[0] == P.shape[1] == X.shape[0]

	plt.show()
