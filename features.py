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

	def extrapolate(self, P: torch.Tensor, X: torch.Tensor, t: int):
		d = X.shape[0]
		P = P.detach()
		Y = torch.full((X.shape[0], t), np.nan, device=X.device)
		Y[:, 0:self.tau+1] = X[:, 0:self.tau+1]
		for i in range(self.tau+1, t):
			z = torch.flatten(Y[:, i-self.tau-1:i].t())
			z = torch.mm(P, z).view(-1)
			Y[:, i] = z[:d]
		return Y

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

	def extrapolate(self, P: torch.Tensor, X: torch.Tensor, t: int):
		P, x = P.detach(), X[:,0].detach().unsqueeze(1)
		Y = torch.full((X.shape[0], t), np.nan, device=X.device)
		for i in range(t):
			if i == 0:
				Y[:, i] = obs.preimage(torch.mm(P, obs(x))).view(-1)
			else:
				x = Y[:, i-1].unsqueeze(1)
				Y[:, i] = obs.preimage(torch.mm(P, obs(x))).view(-1)
		return Y


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
	def __init__(self):
		pass

class GaussianKernel(Kernel):
	def __init__(self, sigma: float):
		self.sigma = sigma
	def gramian(self, X: torch.Tensor, Y: torch.Tensor):
		return torch.exp(-torch.pow(torch.cdist(X, Y, p=2), 2)/(2*self.sigma**2))

class LaplacianKernel(Kernel):
	def __init__(self, sigma: float):
		self.sigma = sigma
	def gramian(self, X: torch.Tensor, Y: torch.Tensor):
		return torch.exp(-torch.cdist(X, Y, p=2)/self.sigma)

class PolyKernel(Kernel):
	def __init__(self, c: float, p: int):
		self.c, self.p = c, p
	def gramian(self, X: torch.Tensor, Y: torch.Tensor):
		return torch.pow(self.c + torch.mm(X.t(), Y), self.p)

'''
Tests
'''

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
	obs = DelayObservable(2)
	X = torch.Tensor([[i*x for x in range(3)] for i in range(1, 6)]).t()
	print('X: ', X)
	Y = obs(X)
	print('Y: ', Y)
	Z = obs.preimage(Y)
	print('Z: ', Z)
	assert (X == Z).all().item(), 'delay preimage incorrect'
