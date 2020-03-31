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

alpha=-1.0
beta=1.0
gamma=0.5
delta=0.3

def solve_mpc(t0: float, dt: float, x0: torch.Tensor, Ps: list, B: torch.Tensor, obs: Observable, cost: Callable, h: int, umin=-2., umax=2., eps=1e-4):
	'''
	h: horizon
	Ps: list of models (if > 1, this does robust MPC)
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

def mpc_loop(x0, y0, Ps, B, obs, cost, h, dt, nmax, tapply=1):
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

if __name__ == '__main__':
	# Init features
	p, d, k = 5, 2, 15
	obs = PolynomialObservable(p, d, k)

	# Try it
	data = hkl.load('saved/duffing_controlled_nominal.hkl')
	P, B = torch.from_numpy(data['P']).float(), torch.from_numpy(data['B']).float()
	dt = data['dt']
	print('Using dt:', dt)

	# xR = lambda t: torch.full(t.shape, 0.)
	# xR = lambda t: torch.sign(torch.cos(t/4))

	# step cost
	lo, hi = -.8, .8
	nstep = 5
	xR = lambda t: torch.floor(t/nstep)*(hi-lo)/nstep + lo
	
	cost = lambda u, x, t: ((x[0] - xR(t))**2).sum()
	h = 50
	x0, y0 = .5, 0.


	hist_t, hist_u, hist_x = mpc_loop(x0, y0, [P], B, obs, cost, h, dt, 1000)

	results = {
		'dt': dt,
		't': hist_t,
		'u': hist_u,
		'x': hist_x,
		'r': xR(torch.Tensor(hist_t)).numpy(),
	}
	print('Saving...')
	hkl.dump(results, 'saved/duffing_mpc.hkl')

