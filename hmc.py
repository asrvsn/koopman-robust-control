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

def hmc_step(params: tuple, momentum: tuple, potential: Callable, n_skip: int, step_size: int):
	def params_grad(p):
		p = tuple(w.detach().requires_grad_() for w in p)
		u = potential(p)
		d_p = torch.autograd.grad(u, p)
		return d_p

	# TODO: check signs here
	momentum = zip_with(momentum, params_grad(params), lambda m, dp: m + 0.5*step_size*dp)

	for n in range(n_skip):
		params = zip_with(params, momentum, lambda p, m: p + step_size*m)
		momentum = zip_with(momentum, params_grad(params), lambda m, dp: m + step_size*dp)

	momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - 0.5*step_size*dp)
	return params, momentum

def accept(h_old: torch.Tensor, h_new: torch.Tensor):
	rho = min(0., h_old - h_new)
	return rho >= torch.log(torch.rand(1).to(h_old.device))

def sample(n_samples: int, init_params: tuple, potential: Callable, step_size=0.03, n_skip=10, n_burn=10):
	params = tuple(x.clone().requires_grad_() for x in init_params)
	ret_params = []
	n = 0
	while len(ret_params) < n_samples:
		momentum = gibbs(params)
		h_old = hamiltonian(params, momentum, potential)
		params, momentum = hmc_step(params, momentum, potential, n_skip, step_size)

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