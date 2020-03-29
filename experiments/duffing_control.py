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

set_seed(1000)

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

# # MPC
# def mpc_solve(P, B, obs, x_init, H=50):
# 	d = x_init.value.shape[0]
# 	d_u = B.shape[1]
# 	x = cp.Parameter((d, H+1))
# 	u = cp.Variable((d_u, H))

# 	Q = np.diag([1.0, 0.0])
# 	R = np.diag([0.01])

# 	cost = 0.
# 	constr = [x[:,0] == x_init]
# 	for t in range(H):
# 		cost += cp.quad_form(x[:, t+1], Q)
# 		# cost += cp.quad_form(u[:, t], R)

# 		z_t = [None for _ in range(obs.k)]
# 		for i, key in enumerate(obs.psi.keys()):
# 			z_t[i] = 1
# 			for term, power in enumerate(key):
# 				if power > 0:
# 					z_t[i] *= x[term, t] ** power
# 		z_t = cp.vstack(z_t)

# 		z_pred = P@z_t + B@(u[:, t][np.newaxis])
# 		x_pred = z_pred[:obs.d]
# 		constr += [x[:, t+1][:, np.newaxis] == x_pred]

# 	prob = cp.Problem(cp.Minimize(cost), constr)
# 	result = prob.solve(verbose=False)
# 	if prob.status == cp.OPTIMAL:
# 		print(result)
# 		ox, ou = x.value[0,:], u.value[0,:]
# 		return 
# 	else:
# 		raise Exception('Problem could not be solved.')

# x_init = cp.Parameter(2)
# x_init.value = np.array([-1., -1.])
# ox, ou = mpc_solve(P.numpy(), B.numpy(), obs, x_init)

# fig, axs = plt.subplots((1,2))
# axs[0].plot(ox[0], ox[1])
# axs[1].plot(ou)

plt.show()
