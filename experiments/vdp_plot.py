import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import hickle as hkl
import random
import torch
import hickle as hkl

method = 'kernel'
results = hkl.load(f'saved/duffing_{method}.hkl')
nominal = torch.from_numpy(results['nominal']).float()
posterior = results['posterior']
samples = [torch.from_numpy(s).float() for s in results['samples']]
step = results['step']
beta = results['beta']

y_samples, x_samples = 2, 6
fig, axs = plt.subplots(y_samples, x_samples)
fig.suptitle(f'Perturbations of VDP oscillator ({method})')

r, c = 0, 0
for perturbed in trajectories:
	ax = axs[r, c]
	xbound, ybound = 7.0, 7.0
	ax.set_xlim(left=-xbound, right=xbound)
	ax.set_ylim(bottom=-ybound, top=ybound)
	for tr in perturbed:
		ax.plot(tr[0], tr[1], color='grey', alpha=0.4)
	c += 1
	if c == x_samples:
		c = 0
		r += 1

plt.figure()
plt.hist(posterior, density=True, bins=int(len(posterior)/4))
plt.title('Posterior distribution')

# Closeup

# t = 300

# plt.figure()
# for Pn in samples:
# 	Zn = obs.extrapolate(Pn.cpu(), X, t+50)
# 	plt.plot(Zn[0], Zn[1], color='grey', alpha=0.3)

# plt.xlim(left=X[0][t], right=X[0][0] + 0.05)
# plt.ylim(bottom=X[1][t], top=X[1][0] + 0.05)
# plt.title(f'beta={beta}, step={hmc_step}, T={T}, distance={"Frobenius" if baseline else "Kernel"}')
# Z0 = obs.extrapolate(nominal.cpu(), X, t+100)
# plt.plot(Z0[0], Z0[1], label='Nominal')

# plt.plot([X[0][0]], [X[1][0]], marker='o', color='blue')
# plt.legend()

plt.show()