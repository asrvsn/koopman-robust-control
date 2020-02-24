'''
Algorithms for computing Perron-Frobenius operator.

PyTorch adaptation of [d3s](https://github.com/sklus/d3s), Klus et al
'''
import torch

from features import *

def edmd(X: torch.Tensor, Y: torch.Tensor, psi: Observable):
	PsiX = psi(X)
	PsiY = psi(Y)
	C_XX = torch.mm(PsiX, PsiX.t())
	C_YX = torch.mm(PsiX, PsiY.t()).t()
	P = torch.mm(torch.pinverse(C_XX), C_YX)
	return P

def kdmd(X: torch.Tensor, Y: torch.Tensor, k: Kernel, epsilon=0):
	n, device = X.shape[0], X.device
	G_XX = gramian(X, X, k)
	G_XY = gramian(X, Y, k).t()
	P = torch.mm(torch.pinverse(G_XX + epsilon*torch.eye(n, device=device)), G_XY)
	return P