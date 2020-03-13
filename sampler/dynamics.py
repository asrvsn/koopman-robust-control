'''
Sample dynamical models around a nominal.
'''
from typing import Callable
import torch

import sampler.hmc as hmc
from utils import *

def perturb(n_samples: int, P0: torch.Tensor, dist_func: Callable, beta: float, \
			sp_div=(1e-3, 1e-3), alpha=1, hmc_step=0.0001, hmc_leapfrog=25, hmc_burn=20, hmc_boundary_resolution=20):
	'''
	n_samples: number of samples to return
	P0: nominal state operator
	dist_func: distance function between state operators
	beta: distribution spread parameter (higher = smaller variance)
	sp_div: (upper, lower) spectral radius deviation allowed from nominal operator
	'''
	dev = P0.device

	eig_max = spectral_radius(P0) 
	s_max, s_min = eig_max + sp_div[0], eig_max - sp_div[1]

	pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(dev), torch.Tensor([beta]).to(dev))

	def potential(params: tuple):
		(P,) = params
		d_k = dist_func(P0, P).clamp(1e-8, 1-1e-8)
		print(d_k.item())
		u = -pdf.log_prob(d_k)
		return u

	def boundary(params: tuple, momentum: tuple, step: float):
		(P,) = params
		(M,) = momentum
		s = spectral_radius(P + step*M)
		if s > s_max:
			micro_step = step/hmc_boundary_resolution
			P_cand = P.detach()
			while spectral_radius(P_cand) <= s_max:
				P_cand += micro_step*M
			P_cand = P_cand.requires_grad_(True)
			# Reflect along plane orthogonal to spectral gradient
			dS = torch.autograd.grad(spectral_radius(P_cand), P_cand)[0]
			M_para = torch.trace(torch.mm(M.t(), dS)) * dS / torch.trace(torch.mm(dS.t(), dS))
			M_refl = M - 2*M_para
			return ((P_cand,), (M_refl,))
		elif s < s_min:
			micro_step = step/hmc_boundary_resolution
			P_cand = P.detach()
			while spectral_radius(P_cand) >= s_min:
				P_cand += micro_step*M
			P_cand = P_cand.requires_grad_(True)
			# Reflect along plane orthogonal to spectral gradient
			dS = -torch.autograd.grad(spectral_radius(P_cand), P_cand)[0]
			M_para = torch.trace(torch.mm(M.t(), dS)) * dS / torch.trace(torch.mm(dS.t(), dS))
			M_refl = M - 2*M_para
			return ((P_cand,), (M_refl,))
		return None

	print('Sampling...')

	samples, ratio = hmc.sample(n_samples, (P0,), potential, boundary, n_burn=hmc_burn, n_leapfrog=hmc_leapfrog, step_size=hmc_step)
	samples = [P for (P,) in samples]

	print('Acceptance ratio: ', ratio)
	print('Nominal spectral radius:', eig_max.item())
	print('Perturbed spectral radii:', [spectral_radius(P).item() for P in samples])

	return samples