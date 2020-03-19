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
		n_subproblems=20, dist_func=None, sp_div=(1e-3, 1e-3), alpha=1., 
		hmc_step=1e-6, hmc_leapfrog=100, hmc_burn=20, 
		hmc_random_step=False, hmc_deterministic=True, debug=False, return_ics=False,
		kernel_m=2, kernel_T=80
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
	return_ics: return initial conditions for parallel HMC
	kernel_m: subspace kernel m 
	kernel_T: subspace kernel T
	'''
	dev = model.device

	if dist_func is None:
		assert len(model.shape) == 2 and model.shape[0] == model.shape[1], "Subspace kernel valid for square matrices only"
		K = PFKernel(dev, model.shape[0], kernel_m, kernel_T)
		dist_func = lambda x, y: K(x, y, normalize=True) 

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

	if return_ics:
		return samples, [ic for (ic,) in initial_conditions]
	else:
		return samples

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import scipy.linalg as linalg 
	import random

	set_seed(9001)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# torch.autograd.set_detect_anomaly(True)

	# Various 2x2 LTI systems 

	systems = {
		'spiral sink': np.array([[-1.,-3.],[2., 1.]]), 
		# 'spiral source': np.array([[-1.,-3.],[2., 1.5]]), 
		# 'nodal source': np.array([[-3.75,-2.15],[2.05, 1.05]]), 
		# 'nodal sink ': np.array([[-0.05,-3.00],[0.35, -3.45]]), 
		# 'center': np.array([[-1.05,-3.60],[1.10, 1.05]]), 
		# 'saddle1': np.array([[0.70,-2.55],[-0.10, -2.50]]), 
		# 'saddle2': np.array([[0.95,2.25],[0.20, -2.10]]), 
	}

	n_perturb = 10
	n_samples = 100
	beta = 5
	sp_div = (1e-3, 1e-3)
	fig, axs = plt.subplots(2 + n_perturb,len(systems))

	for i, (name, A) in enumerate(systems.items()):

		expA = torch.from_numpy(linalg.expm(A)).float()
		samples, ics = perturb(n_samples, expA, beta, sp_div=sp_div, debug=True, return_ics=True)
		samples = [linalg.logm(s.numpy()) for s in samples]
		radii = [spectral_radius(ic).item() for ic in ics]

		# Original phase portrait
		plot_flow_field(axs[0][i], lambda x: A@x, (-4,4), (-4,4))
		axs[0][i].set_title(name)

		# Spectral radii of HMC initial conditions
		axs[1][i].plot(radii, repeat(0))
		axs[1][i].set_title('Initial spectral radii')

		# Perturbed phase portraits
		for j, S in enumerate(random.choices(samples, n_perturb)):
			plot_flow_field(axs[j+2][i], lambda x: S@x, (-4,4), (-4,4))

	fig.suptitle('Perturbations of 2x2 LTI systems')

	plt.show()
