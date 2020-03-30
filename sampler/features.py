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
	def __init__(self, d: int, k: int, m: int):
		'''
		d: input (data) dimension
		k: output (observable) dimension
		m: data length (memory) required for making a single observation 

		Base class is identity observable
		'''
		self.d = d 
		self.k = k 
		self.m = m

	def __call__(self, X: torch.Tensor):
		return X

	def preimage(self, Y: torch.Tensor):
		return Y

	def extrapolate(self, P: torch.Tensor, X: torch.Tensor, t: int, B=None, u=None, unlift_every=True, build_graph=False):
		'''
		P: transfer operator
		X: initial conditions
		t: trajectory length
		B: (optional) control matrix
		u: (optional) control inputs
		'''
		assert X.shape[0] == self.d, "dimension mismatch"
		assert X.shape[1] >= self.m, "insufficient initial conditions provided"
		if not build_graph:
			P, X = P.detach(), X.detach() 
		if u is not None:
			assert B is not None, "Control matrix required"
			assert u.shape[1] >= t, "insufficient control inputs provided"
			if not build_graph:
				B, u = B.detach(), u.detach()

		if unlift_every:
			if build_graph:
				Y = [X[:,0]]
				x_cur = X[:,0].unsqueeze(1)
				for i in range(self.m, t):
					z = P@self(x_cur, build_graph=True)
					if u is not None:
						z = z + B@u[:, i]
					x_cur = self.preimage(z)
					Y.append(x_cur.view(-1))
				return torch.stack(Y, dim=1)
			else:
				Y = torch.full((self.d, t), np.nan, device=X.device)
				Y[:, 0:self.m] = X[:, 0:self.m]
				for i in range(self.m, t):
					x = Y[:, i-self.m:i]
					z = P@self(x)
					if u is not None:
						z += B@u[:, i]
					Y[:, i] = self.preimage(z).view(-1)
				return Y
		# TODO: why would these have any difference?
		else:
			Z = torch.full((self.k, t), np.nan, device=X.device)
			Z[:, 0] = self(X[:, 0:self.m]).view(-1)
			z = Z[:, self.m-1].unsqueeze(1)
			for i in range(self.m, t):
				z = P@z
				if u is not None:
					z = z + B@u[:, i]
				Z[:, i] = z.view(-1)
			return self.preimage(Z)

class ComposedObservable(Observable):
	''' Compose multiple observables (e.g. poly + delay) into a single one '''
	def __init__(self, seq: list):
		assert len(seq) > 0
		self.seq = seq
		d, k, m = seq[0].d, seq[-1].k, np.prod([obs.m for obs in seq])
		super().__init__(d, k, m)

	def __call__(self, X: torch.Tensor):
		Z = X
		for obs in self.seq:
			Z = obs(Z)
		return Z

	def preimage(self, Z: torch.Tensor):
		X = Z
		for obs in reversed(self.seq):
			X = obs.preimage(X)
		return X

class DelayObservable(Observable):
	''' Delay-coordinate embedding '''
	def __init__(self, d: int, tau: int):
		assert tau >= 0
		self.tau = tau
		k = (tau + 1) * d
		m = self.tau + 1
		super().__init__(d, k, m)

	def __call__(self, X: torch.Tensor):
		n = X.shape[1]
		assert n >= self.tau + 1
		Xs = tuple(X[:, i:n-self.tau+i] for i in range(self.tau+1))
		Z = torch.cat(Xs, 0)
		return Z

	def preimage(self, Z: torch.Tensor):
		return Z[:self.d]

class PolynomialObservable(Observable):
	def __init__(self, p: int, d: int, k: int):
		# TODO: if k is too high, this procedure will loop forever

		assert p > 0 and k > 0 
		assert k >= d, "Basis dimension must be at least as large as full state observable"
		self.p = p # max degree of polynomial
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

		super().__init__(d, k, 1)

	def __call__(self, X: torch.Tensor, build_graph=False):
		if build_graph:
			Z = []
			for i, key in enumerate(self.psi.keys()):
				z = torch.ones((X.shape[1],), device=X.device)
				for term, power in enumerate(key):
					if power > 0:
						z = z * torch.pow(X[term], power)
				Z.append(z)
			return torch.stack(Z)
		else:
			Z = torch.empty((self.k, X.shape[1]), device=X.device)
			for i, key in enumerate(self.psi.keys()):
				Z[i] = torch.ones((X.shape[1],), device=X.device)
				for term, power in enumerate(key):
					if power > 0:
						Z[i] *= torch.pow(X[term], power)
			return Z

	def preimage(self, Z: torch.Tensor): 
		return Z[:self.d]

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
	d, tau, n = 3, 3, 6
	obs = DelayObservable(d, tau)
	X = torch.Tensor([[i*x for x in range(d)] for i in range(1, n+1)]).t()
	print('X: ', X)
	Y = obs(X)
	print('Y: ', Y)
	Z = obs.preimage(Y)
	print('Z: ', Z)
	assert (X[:, :Z.shape[1]] == Z).all().item(), 'delay preimage incorrect'

	# print('Composed obs. test')
	# p, d, k, tau = 3, 5, 20, 2
	# obs1 = PolynomialObservable(p, d, k)
	# obs2 = DelayObservable(d, tau)
	# obs = ComposedObservable([obs1, obs2])
	# X = torch.Tensor([[i*x for x in range(d)] for i in range(1, 10)]).t()
	# print(X)
	# Y = obs(X)
	# print(Y)
	# Z = obs.preimage(Y)
	# print(Z)
	# assert (X == Z).all().item(), 'composed preimage incorrect'
