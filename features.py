'''
Features for extended DMD and kernel DMD.

PyTorch adaptation of [d3s](https://github.com/sklus/d3s), Klus et al
'''
from typing import Callable
import torch

'''
Observables
'''

class Observable:
	def __init__(self):
		pass

class MonomialObservable(Observable):
	def __init__(self, p: int):
		self.p = p
	def __call__(self, X: torch.Tensor):
		# TODO
		pass

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