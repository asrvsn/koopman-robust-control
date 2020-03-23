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

def make_ics(N: int, model: torch.Tensor, r_div: tuple, step: float, leapfrog: int, debug=False):
	rad = spectral_radius(model).item()
	r_max, r_min = rad + r_div[0], rad - r_div[1]
	boundary = reflections.fn_boundary(spectral_radius, vmin=r_min, vmax=r_max)
	potential = lambda _: 0 # Uniform 
	ics, ratio = hmc.sample(N, (model,), potential, boundary, step_size=step, n_leapfrog=leapfrog, n_burn=0, random_step=False, return_first=True, debug=debug)
	ics = [ic[0] for ic in ics]
	if debug:
		print('IC acceptance ratio:', ratio)
	return ics

def perturb(
		max_samples: int, model: torch.Tensor, beta: float,
		n_split=20, dist_func=None, r_div=(1e-3, 1e-3), r_step=1e-5, r_leapfrog=100, alpha=1., 
		hmc_step=1e-6, hmc_leapfrog=100, hmc_burn=20, 
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

	if dist_func is None:
		assert len(model.shape) == 2 and model.shape[0] == model.shape[1], "Subspace kernel valid for square matrices only"
		K = PFKernel(dev, model.shape[0], kernel_m, kernel_T)
		dist_func = lambda x, y: K(x, y, normalize=True) 

	# Sample initial conditions uniformly from constraint set 
	print('Generating initial conditions...')
	ics = make_ics(n_split, model, r_div, r_step, r_leapfrog, debug=debug)

	# Combine parallel HMC samples from initial conditions
	print('Sampling models...')
	pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(dev), torch.Tensor([beta]).to(dev))
	def potential(params: tuple):
		d_k = dist_func(model, params[0]).clamp(1e-8)
		return -pdf.log_prob(d_k)

	rad = spectral_radius(model).item()
	r_max, r_min = rad + r_div[0], rad - r_div[1]
	boundary = reflections.fn_boundary(spectral_radius, vmin=r_min, vmax=r_max)

	n_subsamples = int(max_samples / n_split)
	ics = [(ic,) for ic in ics]
	samples = hmc_parallel.sample(n_subsamples, ics, potential, boundary, step_size=hmc_step, n_leapfrog=hmc_leapfrog, n_burn=hmc_burn, random_step=hmc_random_step, return_first=True, debug=debug)
	samples = [s for (s,) in samples]

	return samples

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import scipy.linalg as linalg 
	import random

	from systems.lti2x2 import systems

	set_seed(9001)
	set_mp_backend()
	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	device = 'cpu'
	# torch.autograd.set_detect_anomaly(True)

	K = PFKernel(device, 2, 2, 80)
	dist_func = lambda x, y: K(x, y, normalize=True) 

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

	# Test dynamics sampler
	n_show = 3
	n_samples = 1000
	n_split = 100
	beta = 5
	fig, axs = plt.subplots(3 + n_show, len(systems))

	for i, (name, A) in enumerate(systems.items()):

		expA = torch.from_numpy(linalg.expm(A)).float()
		samples = perturb(n_samples, expA, beta, r_div=(1e-2, 1e-2), r_step=3e-5, debug=True, dist_func=dist_func, n_split=n_split)
		sampledist = [dist_func(expA, s).item() for s in samples]
		samples = [linalg.logm(s.numpy()) for s in samples]

		# Original phase portrait
		plot_flow_field(axs[0,i], lambda x: A@x, (-4,4), (-4,4))
		axs[0,i].set_title(f'{name} - original')

		# Posterior distribution
		axs[1,i].hist(sampledist, density=True, bins=max(1, int(n_samples/4))) 
		axs[1,i].set_title('Posterior distribution')

		# Trace-determinant plot
		tr = [np.trace(S) for S in samples]
		det = [np.linalg.det(S) for S in samples]
		axis = np.linspace(-4, 4, 60)
		axs[2,i].plot(axis, np.zeros(60), color='black')
		axs[2,i].plot(np.zeros(30), np.linspace(0,4,30), color='black')
		axs[2,i].plot(axis, (axis**2)/4, color='black')
		axs[2,i].plot(tr, det, color='blue')
		axs[2,i].plot([np.trace(A)], [np.det(A)], color='orange')
		axs[2,i].set_title('Trace-determinant plane')

		# Perturbed phase portraits
		for j, S in enumerate(random.choices(samples, k=n_show)):
			plot_flow_field(axs[j+3,i], lambda x: S@x, (-4,4), (-4,4))

	fig.suptitle('Perturbations of 2x2 LTI systems')

	plt.show()
