import torch
import hamiltorch
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
		return dist.log_prob(d_pf)
	return log_prob

def hamiltonian(params, momentum, log_prob):
	U = -log_prob(params)
	K = 0.5 * torch.trace(torch.mm(momentum.t(), momentum))
	return U + K 

def gibbs(params):
	dist = torch.distributions.MultivariateNormal(torch.zeros_like(params))
	return dist.sample()

def leapfrog(params, momentum, log_prob, n_skip, step_size):
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

def accept(h_old, h_new):
	rho = min(0., h_old - h_new)
	return rho >= torch.log(torch.rand(1))

def sample(n_samples: int, log_prob: Callable, P0: torch.Tensor, step_size=0.03, n_skip=10, n_burn=10, seed=None):
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

	return list(map(lambda t: t.detach(), ret_params))

# def mk_log_prob(P0: torch.Tensor, K: PFKernel, rate=100.0):
# 	rate = torch.Tensor([rate]).to(P0.device)
# 	dist = torch.distributions.exponential.Exponential(rate)
# 	d = P0.shape[0]
# 	def log_prob(P1: torch.Tensor):
# 		if (torch.isnan(P1).any().item()):
# 			print('a')
# 		d_pf = K(P0, P1.reshape((d,d)), normalize=True)
# 		if (torch.isnan(d_pf).item()):
# 			print('b')
# 		return dist.log_prob(d_pf)
# 	return log_prob


# def sample(n_samples: int, log_prob: Callable, P0: torch.Tensor, step_size=0.03, n_skip=10, seed=None):
# 	if seed is not None:
# 		hamiltorch.set_random_seed(seed)
# 	d = P0.shape[0]
# 	P_init = P0.reshape((d**2,)).clone()
# 	results = hamiltorch.sample(
# 		log_prob_func=log_prob, 
# 		params_init=P_init,
# 		num_samples=n_samples, 
# 		step_size=step_size,  
# 		num_steps_per_sample=n_skip,
# 		# sampler=hamiltorch.Sampler.RMHMC,
# 		# integrator=hamiltorch.Integrator.IMPLICIT,
# 	)
# 	with torch.no_grad():
# 		results = [x.reshape((d, d)) for x in results]
# 		return results

if __name__ == '__main__':
	x = torch.randn((10,1))
	logp = torch.log(pdf(x))
	print(logp)
	dist = torch.distributions.exponential.Exponential(1.0)
	logp = dist.log_prob(x)
	print(logp)