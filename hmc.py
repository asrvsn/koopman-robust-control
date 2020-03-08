import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from utils import *

def hamiltonian(params: tuple, momentum: tuple, potential: Callable):
	U = potential(params)
	K = sum([0.5 * torch.trace(torch.mm(m.t(), m)) for m in momentum])
	return U + K 

def gibbs(params: tuple):
	distributions = tuple(torch.distributions.Normal(torch.zeros_like(w), torch.ones_like(w)) for w in params)
	return tuple(d.sample() for d in distributions)

def leapfrog(params: tuple, momentum: tuple, potential: Callable, boundary: Callable, n_leapfrog: int, step_size: float):
	def params_grad(p):
		p = tuple(w.detach().requires_grad_() for w in p)
		u = potential(p)
		d_p = torch.autograd.grad(u, p)
		return d_p

	momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - 0.5*step_size*dp)

	for n in range(n_leapfrog):

		# Reflect particle until not in violation of boundary 
		bc = boundary(params, momentum, step_size)
		while bc is not None: 
			print('Reflected!')
			(params, momentum) = bc
			bc = boundary(params, momentum, step_size)

		params = zip_with(params, momentum, lambda p, m: p + step_size*m)
		momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - step_size*dp)

	momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - 0.5*step_size*dp)
	# momentum = map(lambda m: -m, momentum)
	return params, momentum

def accept(h_old: torch.Tensor, h_new: torch.Tensor):
	rho = min(0., h_old - h_new)
	return rho >= torch.log(torch.rand(1).to(h_old.device))

def sample(n_samples: int, init_params: tuple, potential: Callable, boundary: Callable, step_size=0.03, n_leapfrog=10, n_burn=10):
	'''
	Leapfrog HMC 

	potential: once-differentiable potential function 
	boundary: boundary condition which returns either None or (boundary position, reflected momentum)
	'''
	params = tuple(x.clone().requires_grad_() for x in init_params)
	ret_params = []
	n = 0
	while len(ret_params) < n_samples:
		momentum = gibbs(params)
		h_old = hamiltonian(params, momentum, potential)
		params, momentum = leapfrog(params, momentum, potential, boundary, n_leapfrog, step_size)

		params = tuple(w.detach().requires_grad_() for w in params)
		h_new = hamiltonian(params, momentum, potential)

		if accept(h_old, h_new) and n > n_burn:
			ret_params.append(params)
			print('Accepted')

		n += 1

	ratio = len(ret_params) / (n - n_burn)
	return ret_params, ratio


if __name__ == '__main__':
	# TODO tests
	pass