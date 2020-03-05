'''
Features for extended DMD and kernel DMD.
'''
import torch
import random
import numpy as np

'''
Observables
'''

class Observable:
	def __init__(self):
		pass

class DelayObservable(Observable):
	''' Delay-coordinate embedding '''
	def __init__(self, tau: int):
		assert tau >= 0
		self.tau = tau

	def __call__(self, X: torch.Tensor):
		assert X.shape[1] >= self.tau + 1
		n = X.shape[1] - self.tau
		d = (self.tau + 1) * X.shape[0]
		Y = torch.empty((d, n), device=X.device)
		for i in range(n):
			Y[:,i] = torch.flatten(X[:,i:i+self.tau+1].t())
		return Y

	def preimage(self, Y: torch.Tensor):
		d = int(Y.shape[0] / (self.tau + 1))
		n = Y.shape[1] + self.tau 
		X = torch.empty((d, n), device=Y.device)
		for i in range(Y.shape[1]):
			X[:, i] = Y[:d, i]
		for j in range(1, self.tau+1):
			i = Y.shape[1] + j - 1
			X[:, i] = Y[j*d:(j+1)*d, -1]
		return X

class PolynomialObservable(Observable):
	def __init__(self, p: int, d: int, k: int):
		# TODO: if k is too high, this procedure will loop forever

		assert p > 0 and k > 0 
		assert k > d, "Basis dimension must be larger than full state observable"
		self.d = d # dimension of input
		self.p = p # max degree of polynomial
		self.k = k # dimension of observable basis
		self.psi = dict()

		# full state observable
		for i in range(d):
			key = [0 for _ in range(i)] + [1] + [0 for _ in range(i+1, d)]
			key = tuple(key)
			self.psi[key] = None

		# add higher-order terms
		terms = [None for _ in range(1, p+1)]
		for i in range(1, p+1):
			channels = np.full((i*d,), np.nan)
			for j in range(d):
				channels[j*i:(j+1)*i] = j
			terms[i-1] = channels

		while len(self.psi) < k:
			deg = random.randint(2, p) # polynomial of random degree
			nonzero_terms = np.random.choice(terms[deg-1], deg)
			key = [0 for _ in range(d)] # exponents of terms
			for term in nonzero_terms:
				key[int(term)] += 1
			key = tuple(key)
			if key not in self.psi: # sample without replacement
				self.psi[key] = None

	def __call__(self, X: torch.Tensor):
		with torch.no_grad():
			Z = torch.empty((self.k, X.shape[1]), device=X.device)
			for i, key in enumerate(self.psi.keys()):
				z = torch.ones((X.shape[1],), device=X.device)
				for term, power in enumerate(key):
					if power > 0:
						z *= torch.pow(X[term], power)
				Z[i] = z
			return Z

	def preimage(self, X: torch.Tensor): 
		return X[:self.d]

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
	print('Poly obs. test')
	p, d, k = 3, 5, 20
	obs = PolynomialObservable(p, d, k)
	print('Polynomial terms:')
	print(obs.psi.keys())
	X = torch.randn((d, 10))
	Y = obs(X)
	Z = obs.preimage(Y)
	assert (X == Z).all().item(), 'poly preimage incorrect'

	print('Delay obs. test')
	obs = DelayObservable(3)
	X = torch.Tensor([[i*x for x in range(3)] for i in range(1, 6)]).t()
	print('X: ', X)
	Y = obs(X)
	print('Y: ', Y)
	Z = obs.preimage(Y)
	print('Z: ', Z)
	assert (X == Z).all().item(), 'delay preimage incorrect'
