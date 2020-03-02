import torch
import matplotlib.pyplot as plt
from typing import Callable

from kernel import *

def exponential_pdf(x: torch.Tensor, rate=1.0):
	return rate * torch.exp(-rate*x)

def mk_log_prob(P0: torch.Tensor, K: PFKernel, rate=1.0):
	rate = torch.Tensor([rate]).to(P0.device)
	dist = torch.distributions.exponential.Exponential(rate)
	def log_prob(P1: torch.Tensor):
		d_pf = K(P0, P1, normalize=True)
		print(d_pf.item())
		return dist.log_prob(d_pf)
	return log_prob

def hamiltonian(params: torch.Tensor, momentum: torch.Tensor, log_prob: Callable):
	U = -log_prob(params)
	K = 0.5 * torch.trace(torch.mm(momentum.t(), momentum))
	return U + K 

def gibbs(params: torch.Tensor):
	dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
	return dist.sample()

def leapfrog(params: torch.Tensor, momentum: torch.Tensor, log_prob: Callable, n_skip: int, step_size: int):
	def params_grad(p):
		p = p.detach().requires_grad_()
		logp = log_prob(p)
		return torch.autograd.grad(logp, p)[0]

	# TODO: check signs here
	momentum += 0.5 * step_size * params_grad(params)

	for n in range(n_skip):
		params = params + step_size * momentum
		momentum += step_size * params_grad(params)

	momentum -= 0.5 * step_size * params_grad(params)
	return params, momentum

def accept(h_old: torch.Tensor, h_new: torch.Tensor):
	rho = min(0., h_old - h_new)
	return rho >= torch.log(torch.rand(1))

def sample(n_samples: int, log_prob: Callable, P0: torch.Tensor, step_size=0.03, n_skip=10, n_burn=10):
	params = P0.clone().requires_grad_()
	ret_params = [params.clone()]
	n = 0
	while len(ret_params) < n_samples:
		momentum = gibbs(params)
		ham = hamiltonian(params, momentum, log_prob)
		leapfrog_params, leapfrog_momentum = leapfrog(params, momentum, log_prob, n_skip, step_size)

		params = leapfrog_params.detach().requires_grad_()
		momentum = leapfrog_momentum
		new_ham = hamiltonian(params, momentum, log_prob)

		if accept(ham, new_ham) and n > n_burn:
			ret_params.append(leapfrog_params)

		n += 1

	ratio = len(ret_params) / n
	return list(map(lambda t: t.detach(), ret_params)), ratio


if __name__ == '__main__':
	# TODO tests
	pass