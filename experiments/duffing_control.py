import torch
import hickle as hkl
import random
import itertools
import cvxpy as cp
import numpy as np

from sampler.features import *
from sampler.operators import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

set_seed(9001)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Initial conditions
t_max = 5
n_data = 8000
n_init = 12
ics = []
x0s = np.linspace(-2.0, 2.0, n_init)
xdot0s = np.linspace(-2.0, 2.0, n_init)
ics = list(itertools.product(x0s, xdot0s))

# Control inputs
gamma = 0.5
n_u = 10
d_u = 1 # one-dimensional input
u = np.outer(np.linspace(-1., 1., n_u), np.ones(n_data)) # Constant inputs
u = torch.from_numpy(u).float()

# Predictor
sys = lambda ic, u: duffing.dataset(t_max, n_data, gamma=gamma, x0=ic[0], xdot0=ic[1], u=lambda _: u[0])
P, B = dmdc(sys, ics, u, obs)

# Test predictor
[x0, xdot0] = random.choice(ics)
u_test = random.choice(u)
X, Y = duffing.dataset(t_max, n_data, gamma=gamma, x0=x0, xdot0=xdot0, u=lambda _: u_test[0])

horizon = 8000
plt.figure()
plt.plot(Y[0], Y[1], label='Actual')
u_input = torch.full((1, horizon), u_test[0]).unsqueeze(2)
Yp = obs.extrapolate(P, X, horizon, B=B, u=u_input, unlift_every=True) # Change this to False; why difference?
plt.plot(Yp[0], Yp[1], label='Predicted')
plt.legend()

# MPC
def mpc_solve(P, B, obs, z0, H=50, umin=-1., umax=1.):
	d_u = B.shape[1]
	z = cp.Variable((obs.k, H+1))
	u = cp.Variable((d_u, H))
	umin = np.full((d_u,), umin)
	umax = np.full((d_u,), umax)

	diag = np.zeros(obs.k)
	diag[0] = 1.
	Q = np.diag(diag)
	R = np.diag([0.01])

	cost = 0.
	constr = [z[:,0] == z0]
	for t in range(H):
		cost += cp.quad_form(z[:, t], Q)
		# cost += cp.quad_form(u[:, t], R)

		z_pred = P@z[:, t] + B@u[:, t]
		constr += [z[:, t+1][:obs.d] == z_pred[:obs.d]]
		constr += [umin <= u[:, t], u[:, t] <= umax]

	prob = cp.Problem(cp.Minimize(cost), constr)
	result = prob.solve(verbose=False)
	if prob.status == cp.OPTIMAL:
		print(result)
		return z.value, u.value
	else:
		raise Exception('Problem could not be solved.')

z0 = cp.Parameter(obs.k)
z0.value = obs(X[:,:3]).numpy()[:, 0]
Z, u = mpc_solve(P.numpy(), B.numpy(), obs, z0)
Xu = obs.preimage(Z) 

fig, axs = plt.subplots(1,2)
axs[0].plot(Xu[0], Xu[1])
axs[0].set_title('Phase space')
axs[1].plot(u[0])
axs[1].set_title('Control signal')

plt.show()
