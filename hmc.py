import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from operators import is_semistable

# gelu = torch.nn.GELU()

def gelu(x):
	return x*(1+torch.erf(x/np.sqrt(2)))/2

def mk_potential(P0: torch.Tensor, kernel: Callable, rate=1.0, bound_spectrum=True, pf_thresh=0.0001):
	rate = torch.Tensor([rate]).to(P0.device)
	dist = torch.distributions.exponential.Exponential(rate)
	sp_bound = torch.Tensor([1.0+pf_thresh]).to(P0.device)
	def potential(P1: torch.Tensor):
		d_k = kernel(P0, P1) 
		print(d_k.item())
		u = -dist.log_prob(d_k)
		if bound_spectrum:
			sp = torch.svd(P1)[1].max().pow(2)
			# u = u + torch.max(sp, sp_bound) - sp_bound
			u = u + gelu(sp-1-pf_thresh)
		return u
	return potential

def hamiltonian(params: torch.Tensor, momentum: torch.Tensor, potential: Callable):
	U = potential(params)
	K = 0.5 * torch.trace(torch.mm(momentum.t(), momentum))
	return U + K 

def gibbs(params: torch.Tensor):
	dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
	return dist.sample()

def hmc_step(params: torch.Tensor, momentum: torch.Tensor, potential: Callable, n_skip: int, step_size: int):
	def params_grad(x):
		x = x.detach().requires_grad_()
		u = potential(x)
		return torch.autograd.grad(u, x)[0]

	# TODO: check signs here
	momentum += 0.5 * step_size * params_grad(params)

	for n in range(n_skip):
		params = params + step_size * momentum
		momentum += step_size * params_grad(params)

	momentum -= 0.5 * step_size * params_grad(params)
	return params, momentum

def accept(h_old: torch.Tensor, h_new: torch.Tensor):
	rho = min(0., h_old - h_new)
	return rho >= torch.log(torch.rand(1).to(h_old.device))

def sample(n_samples: int, potential: Callable, P0: torch.Tensor, step_size=0.03, n_skip=10, n_burn=10, pf_thresh=None):
	params = P0.clone().requires_grad_()
	ret_params = []
	n = 0
	while len(ret_params) < n_samples:
		momentum = gibbs(params)
		h_old = hamiltonian(params, momentum, potential)
		params, momentum = hmc_step(params, momentum, potential, n_skip, step_size)

		params = params.detach().requires_grad_()
		h_new = hamiltonian(params, momentum, potential)

		if (pf_thresh is None or is_semistable(params, eps=pf_thresh)) and accept(h_old, h_new) and n > n_burn:
			ret_params.append(params)
			print('Accepted')

		n += 1

	ratio = len(ret_params) / (n - n_burn)
	return list(map(lambda t: t.detach(), ret_params)), ratio


if __name__ == '__main__':
	# TODO tests
	pass