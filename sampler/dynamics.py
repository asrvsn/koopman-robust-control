'''
Sample dynamical models around a nominal.
'''
from typing import Callable
import torch

import sampler.hmc as hmc
import sampler.hmc_parallel as hmc_parallel
import sampler.reflections as reflections
from sampler.utils import *

def perturb(
		max_samples: int, model: torch.Tensor, beta: float,
		n_subproblems=20, dist_func=None, sp_div=(1e-3, 1e-3), alpha=1., 
		hmc_step=1e-6, hmc_leapfrog=100, hmc_burn=20, 
		hmc_random_step=False, hmc_deterministic=True, debug=False
	):
	'''
	max_samples: number of samples returned <= this
	model: nominal dynamics model
	beta: distribution spread parameter (higher = smaller variance)

	n_subproblems: number of samples from constraint set for parallel HMC
	dist_func: custom distance function between models (default: subspace kernel)
	sp_div: (upper, lower) spectral radius deviation allowed from nominal model
	alpha: leave at 1.
	hmc_step: step size for HMC
	hmc_burn: samples discarded during burn-in phase
	hmc_deterministic: use seed for parallel HMC
	'''

	dev = model.device

	# Constraint set
	eig_max = spectral_radius(model).item()
	s_max, s_min = eig_max + sp_div[0], eig_max - sp_div[1]
	boundary = reflections.fn_boundary(spectral_radius, vmin=s_min, vmax=s_max)

	# Sample initial conditions uniformly from constraint set 
	potential = lambda _: 0 
	sp_step = min(0.3, np.abs(sp_div[0])/10, np.abs(sp_div[1])/10) # Set step as divisor of spectral constraint size
	sp_leapfrog = 30
	sp_burn = 0
	n_conditions = max(1, int(max_samples/n_subproblems))
	initial_conditions = hmc.sample(n_conditions, (model,), potential, boundary, step_size=sp_step, n_leapfrog=sp_leapfrog, n_burn=sp_burn, random_step=False, return_first=True, debug=debug)

	# Combine parallel HMC samples from initial conditions
	pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(dev), torch.Tensor([beta]).to(dev))
	def potential(params: tuple):
		(P,) = params
		d_k = dist_func(P0, P).clamp(1e-8)
		if debug:
			print(d_k.item())
		u = -pdf.log_prob(d_k)
		return u

	n_samples = int(max_samples / n_conditions)
	samples = hmc_parallel.sample(n_samples, initial_conditions, potential, boundary, step_size=hmc_step, n_leapfrog=hmc_leapfrog, n_burn=hmc_burn, random_step=hmc_random_step, return_first=True, debug=debug)
	samples = [s for (s,) in samples]

	print('Nominal spectral radius:', eig_max)
	print('Perturbed spectral radii:', [spectral_radius(s).item() for s in samples])

	return samples

if __name__ == '__main__':
	# TODO tests
	pass