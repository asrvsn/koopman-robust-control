import torch
import hickle as hkl
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
from scipy.integrate import odeint, ode
# matplotlib.use('tkagg')

from sampler.features import *
from sampler.operators import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)
# torch.autograd.set_detect_anomaly(True)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

alpha=-1.0
beta=1.0
gamma=0.5
delta=0.3

def solve_mpc(t0: float, dt: float, x0: torch.Tensor, Ps: list, B: torch.Tensor, obs: Observable, cost: Callable, h: int, umin=-2., umax=2., eps=1e-5):
	'''
	h: horizon
	Ps: list of models
	'''
	u = torch.full((1, h), 0).unsqueeze(2) 
	u = torch.nn.Parameter(u)
	opt = torch.optim.SGD([u], lr=0.1, momentum=0.9)

	window = torch.Tensor([t0 + dt*i for i in range(h)])
	loss, prev_loss = torch.Tensor([float('inf')]), torch.Tensor([0.])

	while torch.abs(loss - prev_loss).item() > eps:
		prev_loss = loss
		loss = torch.max(torch.stack([
			cost(u, obs.extrapolate(P, x0.unsqueeze(1), h, B=B, u=u.clamp(umin, umax), build_graph=True, unlift_every=False), window) for P in Ps
		]))
		# print(loss.item())
		opt.zero_grad()
		loss.backward()
		opt.step()
		u.data = u.data.clamp(umin, umax)

	# print('Final loss:', loss.item())

	return u.data

# def mpc_loop(x0, y0, Ps, B, obs, cost, h, dt, nmax, tapply=1):
# 	history_t = [0]
# 	history_u = [0]
# 	t = 0
# 	for i in tqdm(range(nmax), desc='MPC'):
# 		u_opt = solve_mpc(t, dt, torch.Tensor([x0, y0]), Ps, B, obs, cost, h)[0]
# 		for j in range(tapply):
# 			xdot, ydot = duffing.system((x0, y0), None, alpha, beta, gamma, delta, lambda _: u_opt[j][0])
# 			x0 += dt*xdot
# 			y0 += dt*ydot
# 			t += dt
# 			history_t.append(t)
# 			history_u.append(u_opt[j][0].item())
# 	return np.array(history_t), np.array(history_u)


def mpc_loop(x0, y0, Ps, B, obs, cost, h, dt, nmax, tapply=3):
	history_t = [0]
	history_u = [0]
	history_x = [[x0, y0]]
	t = 0

	r = ode(duffing.system_ode).set_integrator('dop853')
	r.set_initial_value((x0, y0), 0.).set_f_params(alpha, beta, gamma, delta, lambda _: 0)

	for i in tqdm(range(nmax), desc='MPC'):
		u_opt = solve_mpc(t, dt, torch.Tensor([x0, y0]), Ps, B, obs, cost, h)[0]
		for j in range(tapply):
			u_cur = u_opt[j][0]
			r.set_f_params(alpha, beta, gamma, delta, lambda _: u_cur)
			r.integrate(r.t + dt)
			t += dt
			history_t.append(t)
			history_u.append(u_cur.item())
			history_x.append(r.y)
	return np.array(history_t), np.array(history_u), np.array(history_x).T

# Try it
data = hkl.load('saved/duffing_controlled_nominal.hkl')
P, B = torch.from_numpy(data['P']).float(), torch.from_numpy(data['B']).float()
dt = data['dt']
print('Using dt:', dt)

# xR = lambda t: torch.full(t.shape, 0.)
xR = lambda t: torch.sign(torch.cos(t/2))
# xR = lambda t: torch.floor(t/5)/5
cost = lambda u, x, t: ((x[0] - xR(t))**2).sum()
h = 50
x0, y0 = .5, 0.

# # Re-simulate the system with controls 
# hist_t, hist_u = mpc_loop(x0, y0, [P], B, obs, cost, h, dt, 1000)
# def controller(t):
# 	i = int(t/dt)-1
# 	if i < len(hist_u):
# 		return hist_u[i]
# 	return 0
# hist_x = odeint(duffing.system, (x0, y0), hist_t, args=(alpha, beta, gamma, delta, controller)).T

hist_t, hist_u, hist_x = mpc_loop(x0, y0, [P], B, obs, cost, h, dt, 200)

fig, axs = plt.subplots(1, 4)
axs[0].plot(hist_t, hist_x[0], color='blue', label='x1')
axs[0].plot(hist_t, xR(torch.Tensor(hist_t)), color='orange', label='reference')

axs[1].plot(hist_t, hist_x[1])

axs[2].plot(hist_x[0], hist_x[1])

axs[3].plot(hist_t, hist_u)

plt.legend()
# hist = []
# x, y = -1., -1.
# for _ in range(1000):
# 	xdot, ydot = duffing.system((x, y), None, -1., 1., 0.5, 0.3, lambda t: 0)
# 	x += dt * xdot
# 	y += dt * ydot
# 	hist.append([x, y])
# hist = np.array(hist).T
# plt.plot(hist[0], hist[1])


plt.show()

