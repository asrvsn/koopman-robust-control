import torch
import hickle as hkl
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
from scipy.integrate import odeint, ode
# matplotlib.use('tkagg')

import systems.duffing as duffing
from sampler.features import *
from sampler.operators import *
from sampler.kernel import *
from experiments.duffing_mpc import mpc_loop
from experiments.duffing_plot import plot_perturbed, plot_posterior

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)
# torch.autograd.set_detect_anomaly(True)

# Select models
data = hkl.load('saved/duffing_controlled_nominal.hkl')
P, B = torch.from_numpy(data['P']).float(), torch.from_numpy(data['B']).float()
dt = data['dt']
print('Using dt:', dt)

data = hkl.load('saved/duffing_uncertainty_set.hkl')
posterior = data['posterior']
# trajectories = data['trajectories']
samples = [torch.from_numpy(s).float() for s in data['samples']]
beta = data['beta']

K = PFKernel(device, P.shape[0], 2, 80)
dist_func = lambda x, y: K(x, y, normalize=True) 

# Bound set
radius = 0.2
samples = [s for s in samples if dist_func(s, P).item() <= radius]
posterior = [p for p in posterior if p <= radius]

# # Resample trajectories
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)
# n_trajectories = 12
# n_ics = 12
# t = 800
# trajectories = [sample_2d_dynamics(s, obs, t, (-2,2), (-2,2), n_ics, n_ics) for s in random.choices(samples, k=n_trajectories)]

# plot_perturbed(trajectories, 2, 6)
# plot_posterior(posterior, beta)

# Robust MPC
print('Running MPC...')
Ps = random.choices(samples, k=10)
Ps.append(P)

# reference = lambda t: torch.full(t.shape, 0.)
# reference = lambda t: torch.sign(torch.cos(t/4))
# reference = lambda t: torch.floor(t/5)/5

# step cost
def reference(t):
	lo, hi = -.8, .8
	nstep = 5
	return torch.floor(t/nstep)*(hi-lo)/nstep + lo

def cost(u, x, t):
	return ((x[0] - reference(t))**2).sum()

h = 100
n = 100
x0, y0 = 0., 0.


hist_t, hist_u, hist_x = mpc_loop(x0, y0, Ps, B, obs, cost, h, dt, n)

results = {
	'dt': dt,
	't': hist_t,
	'u': hist_u,
	'x': hist_x,
	'r': reference(torch.Tensor(hist_t)).numpy(),
}
print('Saving...')
hkl.dump(results, 'saved/duffing_robust_mpc.hkl')

plt.show()
