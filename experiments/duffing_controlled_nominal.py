import torch
import hickle as hkl
import random
import itertools
import numpy as np

from sampler.features import *
from sampler.kernel import *
from sampler.operators import *
from sampler.utils import *
from sampler.dynamics import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.autograd.set_detect_anomaly(True)

set_seed(9001)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Initial conditions
t_max = 200
n_data = 8000
n_init = 14
ics = []
x0s = np.linspace(-2.4, 2.4, n_init)
xdot0s = np.linspace(-2.4, 2.4, n_init)
ics = list(itertools.product(x0s, xdot0s))

# Control inputs
gamma = 0.5
n_u = 10
d_u = 1 # one-dimensional input
u = np.concatenate((np.zeros(n_u), np.linspace(-1., 1., n_u)))
u = np.outer(u, np.ones(n_data)) # Constant inputs
u = torch.from_numpy(u).float()

# Predictor
print('Building model...')
sys = lambda ic, u: duffing.dataset(t_max, n_data, gamma=gamma, x0=ic[0], xdot0=ic[1], u=lambda _: u[0])
P, B = dmdc(sys, ics, u, obs)

# Trajectories
print('Recording trajectories...')
n_ics = 12
t = 800
trajectories = sample_2d_dynamics(P, obs, t, (-2,2), (-2,2), n_ics, n_ics)


results = {
	'P': P.numpy(),
	'B': B.numpy(),
	'gamma': gamma,
	'trajectories': trajectories,
	'dt': t_max / n_data,
}
print('Saving...')
hkl.dump(results, 'saved/duffing_controlled_nominal.hkl')

# # Test predictor
# [x0, xdot0] = [-1., -1.] # random.choice(ics)
# u_test = -0.5 # random.choice(u)[0]
# X, Y = duffing.dataset(t_max, n_data, gamma=gamma, x0=x0, xdot0=xdot0, u=lambda _: u_test)

# horizon = 8000
# plt.figure()
# plt.plot(Y[0], Y[1], label='Actual')
# u_input = torch.full((1, horizon), u_test).unsqueeze(2)
# Yp = obs.extrapolate(P, X, horizon, B=B, u=u_input) 
# plt.plot(Yp[0], Yp[1], label='Predicted')
# plt.legend()

# plt.show()