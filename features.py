'''
Features for extended DMD and kernel DMD.

PyTorch adaptation of [d3s](https://github.com/sklus/d3s), Klus et al
'''
from typing import Callable
from itertools import combinations
import torch
import numpy as np

'''
Observables
'''

class Observable:
	def __init__(self):
		pass

class PolynomialObservable(Observable):
	def __init__(self, p: int, d: int):
		self.d = d # dimension of input
		self.p = p # degree of polynomial
		self.k = 0 # dimension of observable basis
		self.psi = []
		channels = np.full((p*d,), np.nan)
		for i in range(d):
			channels[i*p:(i+1)*p] = i
		for combo in combinations(channels, p):
		# TODO: fix this
			# print(combo)
			def f(X: torch.Tensor):
				z = torch.ones((X.shape[1],))
				for i in combo:
					z *= X[int(i)]
				return z
			self.psi.append(f)
			self.k += 1

	def __call__(self, X: torch.Tensor):
		Z = torch.empty((self.k, X.shape[1]), device=X.device)
		for i, func in enumerate(self.psi):
			Z[i] = func(X)
		return Z

class GaussianObservable(Observable):
	def __init__(self, sigma: float):
		self.sigma = sigma	
	def __call__(self, X: torch.Tensor):
		#TODO 
		pass

'''
Kernels
'''

class Kernel:
	def __init__(self, name: str):
		self.name = name

class GaussianKernel(Kernel):
	def __init__(self, sigma: float):
		self.sigma = sigma
		super().__init__('gaussian')

class LaplacianKernel(Kernel):
	def __init__(self, sigma: float):
		self.sigma = sigma
		super().__init__('laplacian')

class PolyKernel(Kernel):
	def __init__(self, c: float, p: int):
		self.c, self.p = c, p
		super().__init__('polynomial')

def gramian(X: torch.Tensor, Y: torch.Tensor, k: Kernel):
	if k.name == 'gaussian':
		return torch.exp(-torch.pow(torch.cdist(X, Y, p=2), 2)/(2*k.sigma**2))
	elif k.name == 'laplacian':
		return torch.exp(-torch.cdist(X, Y, p=2)/k.sigma)
	elif k.name == 'polynomial':
		return torch.pow(k.c + torch.mm(X.t(), Y), k.p)
	# default linear kernel
	else: 
		return torch.mm(X.t(), Y)

if __name__ == '__main__': 
	# TODO tests
	pass