import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from sampler.utils import *

def hamiltonian(params: tuple, momentum: tuple, potential: Callable):
	U = potential(params)
	K = sum([0.5 * torch.trace(torch.mm(m.t(), m)) for m in momentum])
	return U + K 

def gibbs(params: tuple):
	distributions = tuple(torch.distributions.Normal(torch.zeros_like(w), torch.ones_like(w)) for w in params)
	return tuple(d.sample() for d in distributions)

def leapfrog(params: tuple, momentum: tuple, potential: Callable, boundary: Callable, n_leapfrog: int, step_size: float, zero_nan=False):
	def params_grad(p):
		p = tuple(w.detach().requires_grad_() for w in p)
		u = potential(p)
		d_p = torch.autograd.grad(u, p)
		if zero_nan: 
			d_p = tuple(zero_if_nan(dw) for dw in d_p)
		return d_p

	momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - 0.5*step_size*dp)

	for n in range(n_leapfrog):

		# Reflect particle until not in violation of boundary 
		bc = boundary(params, momentum, step_size)
		while bc is not None: 
			print('Reflected!')
			(params, momentum) = bc
			bc = boundary(params, momentum, step_size)

		params = zip_with(params, momentum, lambda p, m: p + step_size*m)
		momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - step_size*dp)

	momentum = zip_with(momentum, params_grad(params), lambda m, dp: m - 0.5*step_size*dp)
	# momentum = map(lambda m: -m, momentum)
	return params, momentum

def accept(h_old: torch.Tensor, h_new: torch.Tensor):
	rho = min(0., h_old - h_new)
	return rho >= torch.log(torch.rand(1).to(h_old.device))

def sample(
		n_samples: int, init_params: tuple, potential: Callable, boundary: Callable, 
		step_size=0.03, n_leapfrog=10, n_burn=10, random_step=False, debug=False
	):
	'''
	Leapfrog HMC 

	potential: once-differentiable potential function 
	boundary: boundary condition which returns either None or (boundary position, reflected momentum)
	'''
	params = tuple(x.clone().requires_grad_() for x in init_params)
	ret_params = []
	n = 0
	while len(ret_params) < n_samples:
		momentum = gibbs(params)
		h_old = hamiltonian(params, momentum, potential)

		if random_step:
			eps = torch.normal(step_size, 2*step_size, (1,)).clamp(step_size/10)
		else:
			eps = step_size
		params, momentum = leapfrog(params, momentum, potential, boundary, n_leapfrog, eps)

		params = tuple(w.detach().requires_grad_() for w in params)
		h_new = hamiltonian(params, momentum, potential)

		if accept(h_old, h_new) and n > n_burn:
			ret_params.append(params)
			if debug:
				print('Accepted')

		n += 1

	ratio = len(ret_params) / (n - n_burn)
	ret_params = list(map(lambda p: tuple(map(lambda x: x.detach(), p)), ret_params))
	return ret_params, ratio


if __name__ == '__main__':
	import hamiltorch
	import scipy.stats as stats

	torch.autograd.set_detect_anomaly(True)
	set_seed(9001)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Gaussian
	mean = torch.Tensor([0.,0.,0.])
	var = torch.Tensor([.5,1.,2.])**2
	dist = torch.distributions.MultivariateNormal(mean, torch.diag(var))

	def log_prob(omega):
		return dist.log_prob(omega).sum()

	N = 1000
	step = .3
	L = 5
	burn = 50

	params_init = torch.zeros(3)
	samples = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=N, step_size=step, num_steps_per_sample=L)
	samples = np.array([s.numpy() for s in samples])
	x = np.linspace(-6,6,200)
	fig, axs = plt.subplots(1, 3)
	fig.suptitle(f'hamiltorch result, 3d gaussian, var={var.numpy().tolist()}, step={step}')
	for i in range(3):
		axs[i].hist(samples[:,i],density=True,bins=20)
		axs[i].plot(x, stats.norm.pdf(x, loc=mean[i], scale=var[i]))

	def potential(params):
		return -dist.log_prob(params[0].view(-1)).sum()

	params_init = (torch.zeros((3,1)),)
	boundary = lambda *unused: None
	samples, _ = sample(N, params_init, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn)
	samples = np.array([s.view(-1).numpy() for (s,) in samples]) 
	x = np.linspace(-6,6,200)
	fig, axs = plt.subplots(1, 3)
	fig.suptitle(f'sampler.hmc result, 3d gaussian, var={var.numpy().tolist()}, step={step}')
	for i in range(3):
		axs[i].hist(samples[:,i],density=True,bins=20)
		axs[i].plot(x, stats.norm.pdf(x, loc=mean[i], scale=var[i]))

	# Beta
	alpha = torch.Tensor([1,1,1])
	beta = torch.Tensor([3,5,10])
	dist = torch.distributions.beta.Beta(alpha, beta)

	def log_prob(omega):
		return dist.log_prob(omega.clamp(1e-8)).sum()

	N = 1000
	step = .003 # Lower step and larger L are better for beta (since it is supported on a small interval)
	L = 20
	burn = 0

	params_init = torch.zeros(3)
	samples = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=N, step_size=step, num_steps_per_sample=L)
	samples = np.array([s.numpy() for s in samples])
	x = np.linspace(0,1,100)
	fig, axs = plt.subplots(1, 3)
	fig.suptitle(f'hamiltorch result, 3d beta, beta={beta.numpy().tolist()}, step={step}')
	for i in range(3):
		axs[i].hist(samples[:,i],density=True,bins=20)
		axs[i].plot(x, stats.beta.pdf(x, alpha[i], beta[i]))

	def potential(params):
		return -dist.log_prob(params[0].view(-1).clamp(1e-8)).sum()

	def boundary(params, momentum, step):
		m = momentum[0]
		p = params[0]
		p_min, p_max = p.min(), p.max()
		m_para = torch.zeros_like(m)

		if p_min < 0:
			grad = torch.autograd.grad(p_min, p)[0]
			m_para = m_para + ((m.t()@grad) / (grad.t()@grad)) * grad

		if p_max > 1:
			grad = torch.autograd.grad(p_max, p)[0]
			m_para = m_para + ((m.t()@grad) / (grad.t()@grad)) * grad

		if m_para.sum() > 0:
			return ((p.clamp(0,1),), (m - 2*m_para,))
		return None


	params_init = (torch.zeros((3,1)),)
	samples, _ = sample(N, params_init, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn)
	samples = np.array([s.view(-1).numpy() for (s,) in samples]) 
	x = np.linspace(0,1,100)
	fig, axs = plt.subplots(1, 3)
	fig.suptitle(f'sampler.hmc result, 3d beta, beta={beta.numpy().tolist()}, step={step}')
	for i in range(3):
		axs[i].hist(samples[:,i],density=True,bins=20)
		axs[i].plot(x, stats.beta.pdf(x, alpha[i], beta[i]))

	plt.show()