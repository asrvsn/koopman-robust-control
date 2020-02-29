import torch
import hamiltorch
import matplotlib.pyplot as plt
from typing import Callable

from kernel import *

def mk_log_prob(P0: torch.Tensor, K: PFKernel, rate=100.0):
	rate = torch.Tensor([rate]).to(P0.device)
	dist = torch.distributions.exponential.Exponential(rate)
	d = P0.shape[0]
	def log_prob(P1: torch.Tensor):
		d_pf = K(P0, P1.reshape((d,d)), normalize=True)
		print(d_pf)
		return dist.log_prob(d_pf)
	return log_prob


def sample(n_samples: int, log_prob: Callable, P0: torch.Tensor, step_size=0.3, n_skip=10, seed=None):
	if seed is not None:
		hamiltorch.set_random_seed(seed)
	d = P0.shape[0]
	P_init = P0.reshape((d**2,)).clone()
	results = hamiltorch.sample(
		log_prob_func=log_prob, 
		params_init=P_init,
		num_samples=n_samples, 
		step_size=step_size,  
		num_steps_per_sample=n_skip,
		# sampler=hamiltorch.Sampler.RMHMC,
		# integrator=hamiltorch.Integrator.IMPLICIT,
	)
	with torch.no_grad():
		results = [x.reshape((d, d)) for x in results]
		return results
