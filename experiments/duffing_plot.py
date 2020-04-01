import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import scipy.stats as stats

from sampler.utils import *
from sampler.features import *
from sampler.operators import *

def get_color(z):
	if z[0] > -1.5 and z[0] < -0.2 and z[1] > -0.5 and z[1] < 0.5:
		return 'red'
	elif z[0] > 0.2 and z[0] < 1.5 and z[1] > -0.5 and z[1] < 0.5:
		return 'blue'
	else:
		return 'grey'

def plot_one(trajectory):
	plt.figure()
	xbound, ybound = 2.2, 2.2
	plt.xlim(left=-xbound, right=xbound)
	plt.ylim(bottom=-ybound, top=ybound)
	# plt.axis('off')
	for tr in trajectory:
		plt.plot(tr[0], tr[1], color=get_color(tr[:,-1]))

def plot_perturbed(trajectories, n_y, n_x, xbound=2.2, ybound=2.2):
	fig, axs = plt.subplots(n_y, n_x)
	# fig.suptitle(f'Perturbations of Duffing oscillator ({method})')
	r, c = 0, 0
	for perturbed in trajectories:
		ax = axs[r, c]
		ax.set_xlim(left=-xbound, right=xbound)
		ax.set_ylim(bottom=-ybound, top=ybound)
		ax.axis('off')
		for tr in perturbed:
			ax.plot(tr[0], tr[1], color=get_color(tr[:,-1]))
		c += 1
		if c == n_x:
			c = 0
			r += 1
			if r == n_y: 
				break
	plt.tight_layout()

def plot_posterior(posterior, beta=None):
	plt.figure()
	plt.hist(posterior, density=True, bins=int(len(posterior)/4))

	if beta is not None:
		xaxis = np.linspace(0, 1, 100)
		yaxis = stats.beta.pdf(xaxis, 1., beta)
		plt.plot(xaxis, yaxis)

	plt.title('Posterior distribution')

if __name__ == '__main__':
	set_seed(9001)

	method = 'baseline'
	# name = f'duffing_{method}'
	name = 'duffing_uncertainty_set'

	results = hkl.load(f'saved/{name}.hkl')
	nominal = torch.from_numpy(results['nominal']).float()
	posterior = results['posterior']
	samples = [torch.from_numpy(s).float() for s in results['samples']]
	trajectories = results['trajectories']

	# print('Saving trajectories...')
	# n_perturbed = 21
	# n_ics = 14
	# t = 800
	# p, d, k = 5, 2, 15
	# obs = PolynomialObservable(p, d, k)
	# results['trajectories'] = []
	# for P in random.choices(samples, k=n_perturbed):
	# 	results['trajectories'].append(sample_2d_dynamics(P, obs, t, (-2,2), (-2,2), n_ics, n_ics))
	# hkl.dump(results, f'saved/duffing_{method}.hkl')

	plot_perturbed(trajectories, 3, 7)
	if 'beta' in results:
		plot_posterior(posterior, results['beta'])
	else:
		plot_posterior(posterior)

	# Show nominal
	p, d, k = 5, 2, 15
	obs = PolynomialObservable(p, d, k)
	n_ics = 12
	t = 800
	plot_one(sample_2d_dynamics(nominal, obs, t, (-2,2), (-2,2), n_ics, n_ics))

	plt.show()