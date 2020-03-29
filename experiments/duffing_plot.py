import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from sampler.utils import *
from sampler.features import *
from sampler.operators import *

set_seed(9001)

method = 'kernel'
results = hkl.load(f'saved/duffing_{method}.hkl')
nominal = torch.from_numpy(results['nominal']).float()
posterior = results['posterior']
samples = [torch.from_numpy(s).float() for s in results['samples']]
# trajectories = results['trajectories']

print('Saving trajectories...')
n_perturbed = 12
n_ics = 14
t = 800
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)
results['trajectories'] = []
for P in random.choices(samples, k=n_perturbed):
	results['trajectories'].append(sample_2d_dynamics(P, obs, t, (-2,2), (-2,2), n_ics, n_ics))
hkl.dump(results, f'saved/duffing_{method}.hkl')

# def get_color(z):
# 	if z[0] > -1.5 and z[0] < -0.5 and z[1] > -0.5 and z[1] < 0.5:
# 		return 'red'
# 	elif z[0] > 0.5 and z[0] < 1.5 and z[1] > -0.5 and z[1] < 0.5:
# 		return 'blue'
# 	else:
# 		return 'grey'

# y_samples, x_samples = 2, 6
# fig, axs = plt.subplots(y_samples, x_samples)
# # fig.suptitle(f'Perturbations of Duffing oscillator ({method})')
# r, c = 0, 0
# for perturbed in trajectories:
# 	ax = axs[r, c]
# 	xbound, ybound = 2.2, 2.2
# 	ax.set_xlim(left=-xbound, right=xbound)
# 	ax.set_ylim(bottom=-ybound, top=ybound)
# 	ax.axis('off')
# 	for tr in perturbed:
# 		ax.plot(tr[0], tr[1], color=get_color(tr[:,-1]))
# 	c += 1
# 	if c == x_samples:
# 		c = 0
# 		r += 1

# plt.figure()
# plt.hist(posterior, density=True, bins=int(len(posterior)/4))
# plt.title('Posterior distribution')

# plt.show()