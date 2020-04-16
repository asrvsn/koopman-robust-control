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

method = 'kernel'
results = hkl.load(f'saved/vdp_{method}.hkl')
nominal = torch.from_numpy(results['nominal']).float()
posterior = results['posterior']
samples = [torch.from_numpy(s).float() for s in results['samples']]
step = results['step']
beta = results['beta']
trajectories = results['trajectories']

plot_perturbed(trajectories, 3, 4, xbound=6.4, ybound=6.4)
plot_posterior(posterior, beta)

# # Show nominal
# p, d, k = 4, 2, 8
# obs = PolynomialObservable(p, d, k)
# n_ics = 12
# t = 3000
# plot_one(sample_2d_dynamics(nominal, obs, t, (-2,2), (-2,2), n_ics, n_ics))

plt.show()