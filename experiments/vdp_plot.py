import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import hickle as hkl
import random
import torch
import hickle as hkl

from sampler.utils import *
from sampler.features import *
from sampler.operators import *
from experiments.duffing_plot import plot_one, plot_perturbed, plot_posterior

set_seed(9001)

method = 'baseline'
results = hkl.load(f'saved/vdp_{method}.hkl')
nominal = torch.from_numpy(results['nominal']).float()
posterior = results['posterior']
samples = [torch.from_numpy(s).float() for s in results['samples']]
step = results['step']
beta = results['beta']
# trajectories = results['trajectories']

print(f'Method: {method}, Beta: {beta}, Step: {step}')

plot_posterior(posterior, beta)

def extrapolate(K):
	init = torch.Tensor([[1.], [0.]])
	Z = obs.extrapolate(K, init, 3000)
	return Z.numpy()

plt.figure()
p, d, k = 4, 2, 8
obs = PolynomialObservable(p, d, k)
tr = extrapolate(nominal)
plt.plot(tr[0], tr[1], color='blue')

for K in random.choices(samples, k=20):
	tr = extrapolate(K)
	plt.plot(tr[0], tr[1], color='black', alpha=0.3)

plt.scatter([1], [0], color='blue')

plt.xlim(-4, 4)
plt.ylim(-6, 6)
plt.axis('off')
plt.tight_layout()

plt.show()