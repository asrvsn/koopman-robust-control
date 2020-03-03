import torch
import matplotlib.pyplot as plt
from typing import Callable

from kernel import *

def exponential_pdf(x: torch.Tensor, rate=1.0):
	return rate * torch.exp(-rate*x)

def mk_potential(P0: torch.Tensor, K: PFKernel, rate=1.0, bound_spectrum=True, eps=0.001):
	rate = torch.Tensor([rate]).to(P0.device)
	dist = torch.distributions.exponential.Exponential(rate)
	def potential(P1: torch.Tensor):
		d_pf = K(P0, P1, normalize=True)
		print(d_pf.item())
		u = dist.log_prob(d_pf)
		if bound_spectrum:
			u = u + torch.sigmoid(torch.eig(P1)[0].max() - 1.0 - eps)
		return u
	return potential

def hamiltonian(params: torch.Tensor, momentum: torch.Tensor, log_prob: Callable):
	U = -log_prob(params)
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

def sample(n_samples: int, potential: Callable, P0: torch.Tensor, step_size=0.03, n_skip=10, n_burn=10, pf_validate=False):
	params = P0.clone().requires_grad_()
	ret_params = [params.clone()]
	n = 0
	while len(ret_params) < n_samples:
		momentum = gibbs(params)
		h_old = hamiltonian(params, momentum, potential)
		params, momentum = hmc_step(params, momentum, potential, n_skip, step_size)

		params = params.detach().requires_grad_()
		h_new = hamiltonian(params, momentum, potential)

		if (not pf_validate or PFKernel.validate(params, eps=0.001)) and accept(h_old, h_new) and n > n_burn:
			ret_params.append(params)
			print('Accepted')

		n += 1

	ratio = len(ret_params) / (n - n_burn)
	return list(map(lambda t: t.detach(), ret_params)), ratio


if __name__ == '__main__':
	# TODO tests
	pass