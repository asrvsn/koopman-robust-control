'''
Sample dynamical models around a nominal.
'''
from typing import Callable
from itertools import repeat
import torch

import sampler.hmc as hmc
import sampler.hmc_parallel as hmc_parallel
import sampler.reflections as reflections
from sampler.kernel import *
from sampler.utils import *


def perturb(
		max_samples: int, model: torch.Tensor, beta: float,
		n_split=20, dist_func=None, boundary=None,
		r_div=(1e-2, 1e-2), ic_step=1e-5, ic_leapfrog=100, alpha=1., 
		hmc_step=1e-6, hmc_leapfrog=100, hmc_burn=0, 
		hmc_random_step=False, hmc_deterministic=True, debug=False, 
		kernel_m=2, kernel_T=80
	):
	'''
	max_samples: number of samples returned <= this
	model: nominal dynamics model
	beta: distribution spread parameter (higher = smaller variance)

	n_split: number of samples from constraint set for parallel HMC
	dist_func: custom distance function between models (default: subspace kernel)
	r_div: (upper, lower) spectral radius deviation allowed from nominal model
	alpha: leave at 1.
	hmc_step: step size for HMC
	hmc_burn: samples discarded during burn-in phase
	hmc_deterministic: use seed for parallel HMC
	kernel_m: subspace kernel m 
	kernel_T: subspace kernel T
	'''
	dev = model.device
	n_split = min(max_samples, n_split)

	# If no boundary provided, default to spectral radius bound
	if boundary is None:
		rad = spectral_radius(model).item()
		r_max, r_min = rad + r_div[0], rad - r_div[1]
		boundary = reflections.fn_boundary(spectral_radius, vmin=r_min, vmax=r_max)

	# If no distance provided, default to subspace kernel
	if dist_func is None:
		assert len(model.shape) == 2 and model.shape[0] == model.shape[1], "Subspace kernel valid for square matrices only"
		K = PFKernel(dev, model.shape[0], kernel_m, kernel_T)
		dist_func = lambda x, y: K(x, y, normalize=True) 

	# Sample initial conditions uniformly from constraints 
	print('Generating initial conditions...')
	potential = lambda _: 0 # Uniform 
	ics, ratio = hmc.sample(n_split, (model,), potential, boundary, step_size=ic_step, n_leapfrog=ic_leapfrog, n_burn=0, random_step=False, return_first=True, debug=debug)
	if debug:
		print('IC acceptance ratio:', ratio)

	# Run parallel HMC on initial conditions
	print('Sampling models...')
	pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(dev), torch.Tensor([beta]).to(dev))
	def potential(params: tuple):
		d_k = dist_func(model, params[0]).clamp(1e-8)
		return -pdf.log_prob(d_k)

	n_subsamples = int(max_samples / n_split)
	samples = hmc_parallel.sample(n_subsamples, ics, potential, boundary, step_size=hmc_step, n_leapfrog=hmc_leapfrog, n_burn=hmc_burn, random_step=hmc_random_step, return_first=True, debug=debug)
	samples = [s for (s,) in samples]
	posterior = [dist_func(model, s).item() for s in samples]

	return samples, posterior

# if __name__ == '__main__':
	# import matplotlib.pyplot as plt
	# import scipy.linalg as linalg 
	# import random

	# from systems.lti2x2 import systems

	# set_seed(9001)
	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# # torch.autograd.set_detect_anomaly(True)

	# # Test initial conditions 
	# model = linalg.expm(systems['spiral sink'])
	# model = torch.from_numpy(model).float().to(device)
	# N = 100
	# ics = make_ics(N, model, (1e-2, 1e-2), 3e-5, 100, debug=True)
	# ds = [dist_func(model, ic).item() for ic in ics]
	# fig, axs = plt.subplots(1, 2)
	# radii = [spectral_radius(ic).item() for ic in ics]
	# axs[0].scatter(radii, [0 for _ in range(N)])
	# axs[0].set_title('Spectral radii')
	# axs[1].hist(ds, density=True, bins=int(N/3))
	# axs[1].set_title('Distribution')
	# fig.suptitle('Initial conditions from constraint set')

	# plt.show()
