import torch
import hickle as hkl
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('tkagg')

from sampler.features import *
from sampler.operators import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)
# torch.autograd.set_detect_anomaly(True)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

def solve_mpc(t0: float, dt: float, x0: torch.Tensor, P: torch.Tensor, B: torch.Tensor, obs: Observable, cost: Callable, h: int, umin=-1., umax=1., eps=1e-5):
	'''
	h: horizon
	'''
	u = torch.full((1, h), 0).unsqueeze(2) 
	u = torch.nn.Parameter(u)
	opt = torch.optim.SGD([u], lr=0.1, momentum=0.9)

	window = torch.Tensor([t0 + dt*i for i in range(h)])
	loss, prev_loss = torch.Tensor([float('inf')]), torch.Tensor([0.])

	while torch.abs(loss - prev_loss).item() > eps:
		x_pred = obs.extrapolate(P, x0.unsqueeze(1), h, B=B, u=u, build_graph=True, unlift_every=False)
		prev_loss = loss
		loss = cost(u, x_pred, window)
		# print(loss.item())
		opt.zero_grad()
		loss.backward()
		opt.step()
		u.data = u.data.clamp(umin, umax)

	# print('Final loss:', loss.item())

	return u.data

def mpc_loop(x0, y0, P, B, obs, cost, h, dt, nmax, tapply=10):
	history_t = [0]
	history_x = [[x0, y0]]
	history_u = [0]
	t = 0
	for i in tqdm(range(nmax), desc='MPC'):
		u_opt = solve_mpc(t, dt, torch.Tensor([x0, y0]), P, B, obs, cost, h)[0]
		for j in range(tapply):
			xdot, ydot = duffing.system((x0, y0), None, -1., 1., 0.5, 0.3, lambda _: u_opt[j][0])
			x0 += dt*xdot
			y0 += dt*ydot
			t += dt
			history_t.append(t)
			history_x.append([x0, y0])
			history_u.append(u_opt[j][0].item())
	return np.array(history_t), np.array(history_x).T, np.array(history_u)

# Try it
data = hkl.load('saved/duffing_controlled_nominal.hkl')
P, B = torch.from_numpy(data['P']).float(), torch.from_numpy(data['B']).float()
dt = data['dt']
print('Using dt:', dt)

# xR = lambda t: torch.full(t.shape, -0.5)
xR = lambda t: torch.cos(t*0.5)
cost = lambda u, x, t: ((x[0] - xR(t))**2).sum()
h = 50
x0, y0 = 1., 1.

hist_t, hist_x, hist_u = mpc_loop(x0, y0, P, B, obs, cost, h, dt, 100)
plt.figure()
plt.plot(hist_t, hist_x[0], color='blue')
plt.plot(hist_t, xR(torch.Tensor(hist_t)), color='orange')

plt.figure()
plt.plot(hist_t, hist_u)

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

