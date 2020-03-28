import torch
import hickle as hkl
import random

from sampler.features import *
from sampler.operators import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set_seed(9001)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Initial conditions
t_max = 10
n_data = 8000
n_init = 5
x0s = np.linspace(-2.0, 2.0, n_init)
xdot0s = np.linspace(-2.0, 2.0, n_init)
ics = np.vstack((x0s, xdot0s)).T

# Control inputs
gamma = 0.5
n_u = 10
u = np.outer(np.linspace(-1, 1, n_u), np.ones(n_data)) # Constant inputs
u = torch.from_numpy(u).float()

# Predictor
sys = lambda ic, u: duffing.dataset(t_max, n_data, gamma=gamma, x0=ic[0], xdot0=ic[1], u=lambda _: u[0])
P, B = dmdc(sys, ics, u, obs)

# Test predictor
[x0, xdot0] = random.choice(ics)
u_test = random.choice(u)
X, Y = duffing.dataset(t_max, n_data, gamma=gamma, x0=x0, xdot0=xdot0, u=lambda _: u_test[0])

plt.figure()
plt.plot(Y[0], Y[1])
plt.title('Actual')

plt.figure()
u_input = torch.full((1,n_data), u_test[0]).unsqueeze(2)
Yp = obs.extrapolate(P, X, n_data, B=B, u=u_input)
plt.plot(Yp[0], Yp[1])
plt.title('Predicted')

plt.show()