from typing import Callable
from itertools import repeat
import multiprocessing
import torch

from sampler.utils import *
import sampler.hmc as hmc
import sampler.reflections as reflections

# Multiprocessing cannot pickle lambdas
_implicit_potential = None
_implicit_boundary = None

def worker(
		n_samples: int, init_params: tuple, 
		step_size: float, n_leapfrog: int, n_burn: int, random_step: bool, debug: bool, return_first: bool,
		max_refl: int
	):
	samples, ratio = hmc.sample(n_samples, init_params, _implicit_potential, _implicit_boundary, step_size=step_size, n_leapfrog=n_leapfrog, n_burn=n_burn, random_step=random_step, debug=debug, return_first=return_first, max_refl=max_refl)
	print(f'Ratio: {ratio}')
	return samples

def sample(
		n_samples: int, initial_conditions: list, potential: Callable, boundary: Callable,
		step_size=0.03, n_leapfrog=10, n_burn=10, random_step=False, debug=False, return_first=False, 
		max_refl=100
	):

	global _implicit_potential, _implicit_boundary
	_implicit_potential = potential
	_implicit_boundary = boundary

	with multiprocessing.Pool(processes=4) as pool:
		args = zip(
			repeat(n_samples), initial_conditions, 
			repeat(step_size), repeat(n_leapfrog), repeat(n_burn), repeat(random_step), repeat(debug), repeat(return_first),
			repeat(max_refl)
		)
		samples = pool.starmap(worker, args)

	samples = [s for subset in samples for s in subset]
	return samples

if __name__ == '__main__':
	# torch.autograd.set_detect_anomaly(True)
	set_seed(9001)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Reflection test
	N = 20
	step = 0.3
	L = 5
	burn = 2

	pdf = torch.distributions.uniform.Uniform(torch.Tensor([[-1.0], [-1.0]]), torch.Tensor([[1.0], [1.0]]))
	initial_conditions = [(pdf.sample(),) for _ in range(5)]
	potential = lambda _: 0 # uniform over [-1,1]x[-1,1]
	boundary = reflections.rect_boundary(vmin=-1, vmax=1)

	print('Sampling')
	samples = sample(N, initial_conditions, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn, debug=True)
	samples = np.array([s.view(-1).numpy() for (s,) in samples])
	initial_conditions = np.array([s.view(-1).numpy() for (s,) in initial_conditions])

	plt.figure()
	plt.title(f'Parallel HMC on the square, step={step}')
	plt.scatter(initial_conditions[:,0], initial_conditions[:,1], color='orange', label='Initial condition')
	plt.scatter(samples[:,0], samples[:,1])
	x = np.linspace(-1,1,20)
	ones = np.ones(x.shape)
	plt.plot(x,ones,color='black')
	plt.plot(ones,x,color='black')
	plt.plot(x,-ones,color='black')
	plt.plot(-ones,x,color='black')
	plt.legend()

	plt.show()

