import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from sampler.utils import *
from sampler.features import *

# Initial conditions
n_init = 12
x0s = np.linspace(-2.0, 2.0, n_init)
xdot0s = np.linspace(-2.0, 2.0, n_init)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Plot
method = 'kernel'
results = hkl.load(f'saved/duffing_{method}.hkl')
nominal = torch.from_numpy(results['nominal']).float()
posterior = results['posterior']
samples = [torch.from_numpy(s).float() for s in results['samples']]

t = 800

def get_label(z):
	if z[0] > -1.5 and z[0] < -0.5 and z[1] > -0.5 and z[1] < 0.5:
		return 'Left', 'red'
	elif z[0] > 0.5 and z[0] < 1.5 and z[1] > -0.5 and z[1] < 0.5:
		return 'Right', 'blue'
	else:
		return 'Unstable', 'grey'

plt.figure()
xbound, ybound = 2.2, 2.2
plt.xlim(left=-xbound, right=xbound)
plt.ylim(bottom=-ybound, top=ybound)
plt.title('Nominal Duffing oscillator')
for x0 in x0s:
	for xdot0 in xdot0s:
		x = torch.Tensor([[x0], [xdot0]])
		Z0 = obs.extrapolate(nominal, x, t).numpy()
		label, color = get_label(Z0[:, -1])
		plt.plot(Z0[0], Z0[1], color=color)

y_samples, x_samples = 3, 7

fig, axs = plt.subplots(y_samples, x_samples)
fig.suptitle(f'Perturbations of Duffing oscillator ({method})')
r, c = 0, 0
for perturbed in random.choices(samples, k=y_samples*x_samples):
	ax = axs[r, c]
	xbound, ybound = 2.2, 2.2
	ax.set_xlim(left=-xbound, right=xbound)
	ax.set_ylim(bottom=-ybound, top=ybound)
	for x0 in x0s:
		for xdot0 in xdot0s:
			x = torch.Tensor([[x0], [xdot0]])
			Zn = obs.extrapolate(perturbed, x, t).numpy()
			label, color = get_label(Zn[:, -1])
			ax.plot(Zn[0], Zn[1], color=color)
	c += 1
	if c == x_samples:
		c = 0
		r += 1

plt.figure()
plt.hist(posterior, density=True, bins=int(len(posterior)/4))
plt.title('Posterior distribution')

plt.show()