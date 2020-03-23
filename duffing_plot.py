import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from sampler.utils import *

# Initial conditions
n_init = 12
x0s = np.linspace(-2.0, 2.0, n_init)
xdot0s = np.linspace(-2.0, 2.0, n_init)

# Plot
results = hkl.load('saved/duffing.hkl')
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
plt.title('Nominal extrapolation of the Duffing Oscillator')
for x0 in x0s:
	for xdot0 in xdot0s:
		x = torch.Tensor([[x0], [xdot0]])
		Z0 = obs.extrapolate(nominal, x, t)
		label, color = get_label(Z0[:, -1]).numpy()
		if label != 'Unstable':
			plt.plot(Z0[0], Z0[1], label=label, color=color)

deduped_legend()

y_samples, x_samples = 3, 7

fig, axs = plt.subplots(y_samples, x_samples)
fig.suptitle(f'Perturbed extrapolations of the Duffing oscillator')
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
			if label != 'Unstable':
				ax.plot(Zn[0], Zn[1], label=label, color=color)
	c += 1
	if c == x_samples:
		c = 0
		r += 1

deduped_legend()

plt.figure()
plt.hist(posterior, density=True, bins=int(len(posterior)/4))
plt.title('Posterior distribution')

plt.show()