from typing import Callable, Any
from itertools import repeat
import multiprocessing
import torch

from sampler.utils import *
import sampler.hmc as hmc

# Multiprocessing cannot pickle lambdas
_implicit_potential = None
_implicit_boundary = None

def worker(
		n_samples: int, init_params: tuple, 
		step_size: float, n_leapfrog: int, n_burn: int, random_step: bool, debug: bool, return_first: bool,
		seed: Any
	):	
	if seed is not None:
		set_seed(seed)
	samples, ratio = hmc.sample(n_samples, init_params, _implicit_potential, _implicit_boundary, step_size=step_size, n_leapfrog=n_leapfrog, n_burn=n_burn, random_step=random_step, debug=debug, return_first=return_first)
	print(f'Ratio: {ratio}')
	return samples

def sample(
		n_samples: int, initial_conditions: list, potential: Callable, boundary: Callable,
		step_size=0.03, n_leapfrog=10, n_burn=10, random_step=False, debug=False, return_first=False, 
		deterministic=True
	):

	global _implicit_potential, _implicit_boundary
	_implicit_potential = potential
	_implicit_boundary = boundary

	n_problems = len(initial_conditions)
	with multiprocessing.Pool() as pool:
		seeds = range(1000, 1000+n_problems) if deterministic else repeat(None)
		args = zip(
			repeat(n_samples), initial_conditions, 
			repeat(step_size), repeat(n_leapfrog), repeat(n_burn), repeat(random_step), repeat(debug), repeat(return_first),
			seeds
		)
		samples = pool.starmap(worker, args)

	samples = [s for subset in samples for s in subset]
	return samples

if __name__ == '__main__':
	import scipy.stats as stats
	import sampler.reflections as reflections

	# torch.autograd.set_detect_anomaly(True)
	set_seed(9001)
	set_mp_backend()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Reflection test
	N = 20
	step = 0.3
	L = 5
	burn = 2

	pdf = torch.distributions.uniform.Uniform(torch.Tensor([[-1.0], [-1.0]]), torch.Tensor([[1.0], [1.0]]))
	initial_conditions = [(pdf.sample(),) for _ in range(5)]
	potential = lambda _: 0 # uniform over [-1,1]x[-1,1]
	boundary = reflections.lp_boundary(float('inf'), vmax=1)

	samples = sample(N, initial_conditions, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn, deterministic=True)
	samples = np.array([s.view(-1).numpy() for (s,) in samples])
	initial_conditions = np.array([s.view(-1).numpy() for (s,) in initial_conditions])

	plt.figure()
	plt.title(f'sampler.hmc_parallel on the square, step={step}')
	plt.scatter(initial_conditions[:,0], initial_conditions[:,1], color='orange', label='Initial condition')
	plt.scatter(samples[:,0], samples[:,1])
	x = np.linspace(-1,1,20)
	ones = np.ones(x.shape)
	plt.plot(x,ones,color='black')
	plt.plot(ones,x,color='black')
	plt.plot(x,-ones,color='black')
	plt.plot(-ones,x,color='black')
	plt.legend()

	# Gaussian test
	mean = torch.Tensor([0.,0.,0.])
	var = torch.Tensor([.5,1.,2.])**2
	pdf = torch.distributions.MultivariateNormal(mean, torch.diag(var))

	def potential(params):
		return -pdf.log_prob(params[0].view(-1)).sum()

	N = 1000
	n_problems = 20
	step = .3
	L = 5
	burn = 50

	initial_conditions = [(torch.zeros((3,1)),) for _ in range(n_problems)]
	boundary = reflections.nil_boundary
	samples = sample(N, initial_conditions, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn, deterministic=True)
	samples = np.array([s.view(-1).numpy() for (s,) in samples]) 
	x = np.linspace(-6,6,200)
	fig, axs = plt.subplots(1, 3)
	fig.suptitle(f'sampler.hmc_parallel on 3d gaussian, var={var.numpy().tolist()}, step={step}')
	for i in range(3):
		axs[i].hist(samples[:,i],density=True,bins=20)
		axs[i].plot(x, stats.norm.pdf(x, loc=mean[i], scale=var[i]))

	# # Beta test
	# alpha = torch.Tensor([[1.],[1.],[1.]])
	# beta = torch.Tensor([[3.],[5.],[10.]])
	# pdf = torch.distributions.beta.Beta(alpha, beta)
	# init_pdf = torch.distributions.uniform.Uniform(torch.Tensor([[0.0], [0.0], [0.0]]), torch.Tensor([[1.0], [1.0], [1.0]]))

	# def potential(params):
	# 	return -pdf.log_prob(params[0].clamp(1e-8)).sum()

	# N = 100
	# n_problems = 10
	# step = .003 # Lower step and larger L are better for beta (since it is supported on a small interval)
	# L = 20
	# burn = 20

	# initial_conditions = [(init_pdf.sample(),) for _ in range(n_problems)]
	# boundary = reflections.rect_boundary(vmin=0., vmax=1.)
	# samples = sample(N, initial_conditions, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn, deterministic=True)
	# samples = np.array([s.view(-1).numpy() for (s,) in samples]) 
	# x = np.linspace(0,1,100)
	# fig, axs = plt.subplots(1, 3)
	# fig.suptitle(f'sampler.hmc result, 3d beta, beta={beta.numpy().tolist()}, step={step}')
	# for i in range(3):
	# 	axs[i].hist(samples[:,i],density=True,bins=20)
	# 	axs[i].plot(x, stats.beta.pdf(x, alpha[i], beta[i]))

	plt.show()

