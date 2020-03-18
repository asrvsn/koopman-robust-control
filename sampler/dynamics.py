'''
Sample dynamical models around a nominal.
'''
from typing import Callable
import torch

import sampler.hmc as hmc
import sampler.hmc_nuts as hmc_nuts
from sampler.utils import *

def perturb(
			n_samples: int, P0: torch.Tensor, dist_func: Callable, beta: float, \
			sp_div=(1e-3, 1e-3), alpha=1, \
			hmc_step=0.0001, hmc_random_step=False, hmc_leapfrog=25, hmc_burn=20, hmc_boundary_resolution=20, \
			hmc_target_accept=0.75, NUTS=False 
		):
	'''
	n_samples: number of samples to return
	P0: nominal state operator
	dist_func: distance function between state operators
	beta: distribution spread parameter (higher = smaller variance)
	sp_div: (upper, lower) spectral radius deviation allowed from nominal operator

	alpha: leave at 1.
	hmc_step: step size for HMC, initial step size for NUTS HMC
	hmc_burn: samples discarded during burn-in phase
	hmc_boundary_resolution: step-size divisor for identifying boundary locations
	hmc_target_accept: target accept rate for NUTS
	NUTS: use No U-Turn Sampler 
	'''
	dev = P0.device

	eig_max = spectral_radius(P0) 
	s_max, s_min = eig_max + sp_div[0], eig_max - sp_div[1]
	print('Nominal radius:', eig_max.item())

	pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(dev), torch.Tensor([beta]).to(dev))

	def potential(params: tuple):
		(P,) = params
		d_k = dist_func(P0, P).clamp(1e-8, 1)
		print(d_k.item())
		u = -pdf.log_prob(d_k)
		return u

	def boundary(params: tuple, momentum: tuple, step: float):
		(P,) = params
		(M,) = momentum
		s = spectral_radius(P + step*M)
		if s > s_max:
			# print('Upper boundary')
			micro_step = step/hmc_boundary_resolution
			P_cand = P.detach()
			while spectral_radius(P_cand) <= s_max:
				# print('Upper reflection:', spectral_radius(P_cand).item())
				P_cand += micro_step*M
			P_cand = P_cand.requires_grad_(True)
			# Reflect along plane orthogonal to spectral gradient
			dS = torch.autograd.grad(spectral_radius(P_cand), P_cand)[0]
			M_para = (torch.trace(M.t()@dS) / torch.trace(dS.t()@dS)) * dS 
			M_refl = M - 2*M_para
			return ((P_cand,), (M_refl,))
		elif s < s_min:
			# print('Lower boundary')
			micro_step = step/hmc_boundary_resolution
			P_cand = P.detach()
			while spectral_radius(P_cand) >= s_min:
				# print('Lower reflection:', spectral_radius(P_cand).item())
				P_cand += micro_step*M
			P_cand = P_cand.requires_grad_(True)
			# Reflect along plane orthogonal to spectral gradient
			dS = -torch.autograd.grad(spectral_radius(P_cand), P_cand)[0]
			M_para = (torch.trace(M.t()@dS) / torch.trace(dS.t()@dS)) * dS 
			M_refl = M - 2*M_para
			return ((P_cand,), (M_refl,))
		return None

	print('Sampling...')

	if NUTS: 
		samples, ratio, step = hmc_nuts.sample(n_samples, (P0,), potential, boundary, n_burn=hmc_burn, n_leapfrog=hmc_leapfrog, step_size_init=hmc_step, target_accept_rate=hmc_target_accept)
		print('Final step size:', step)
	else:
		samples, ratio = hmc.sample(n_samples, (P0,), potential, boundary, n_burn=hmc_burn, n_leapfrog=hmc_leapfrog, step_size=hmc_step, random_step=hmc_random_step)
	samples = [P for (P,) in samples]

	print('Acceptance ratio: ', ratio)
	print('Nominal spectral radius:', eig_max.item())
	print('Perturbed spectral radii:', [spectral_radius(P).item() for P in samples])

	return samples