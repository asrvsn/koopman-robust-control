from typing import Callable
import torch
import concurrent.futures

from sampler.utils import *
import sampler.hmc as hmc
import sampler.reflections as reflections

def sample(
		n_samples: int, initial_conditions: list, potential: Callable, boundary: Callable,
		step_size=0.03, n_leapfrog=10, n_burn=10, random_step=False, debug=False, return_first=False, 
		max_refl=100
	):
	all_samples = []

	def op(id: int, init_params: tuple):
		print('Bullshit')
		samples, ratio = hmc.sample(n_samples, init_params, potential, boundary, step_size=step_size, n_leapfrog=n_leapfrog, n_burn=n_burn, random_step=random_step, debug=debug, return_first=return_first, max_refl=max_refl)
		print(f'Worker {id} ratio: {ratio}')
		return samples

	with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
		workers = [executor.submit(op, i, ic) for i, ic in enumerate(initial_conditions)]
		for finished_worker in concurrent.futures.as_completed(workers):
			samples = finished_worker.result()
			print('Finished')
			all_samples.extend(samples)

	return all_samples


if __name__ == '__main__':
	# torch.autograd.set_detect_anomaly(True)
	set_seed(9001)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Reflection test
	N = 20
	step = 0.3
	L = 10
	burn = 10

	pdf = torch.distributions.uniform.Uniform(torch.Tensor([[-1.0], [-1.0]]), torch.Tensor([[1.0], [1.0]]))
	initial_conditions = [(pdf.sample(),) for _ in range(10)]
	potential = lambda _: 0 # uniform over [-1,1]x[-1,1]
	boundary = reflections.rect_boundary(vmin=-1, vmax=1)

	print('Sampling')
	samples, _ = sample(N, initial_conditions, potential, boundary, step_size=step, n_leapfrog=L, n_burn=burn, debug=True)
	samples = np.array([s.view(-1).numpy() for (s,) in samples])
	initial_conditions = np.array([s.view(-1).numpy() for (s,) in initial_conditions])

	plt.figure()
	plt.title(f'Parallel HMC on the square, step={step}')
	plt.scatter(initial_conditions[:,0], initial_conditions[:,1], color='orange')
	plt.scatter(samples[:,0], samples[:,1])
	x = np.linspace(-1,1,20)
	ones = np.ones(x.shape)
	plt.plot(x,ones,color='black')
	plt.plot(ones,x,color='black')
	plt.plot(x,-ones,color='black')
	plt.plot(-ones,x,color='black')

	plt.show()

