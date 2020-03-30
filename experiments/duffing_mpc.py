import torch
import hickle as hkl
import random
import numpy as np
import matplotlib.pyplot as plt

from sampler.features import *
from sampler.operators import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)
# torch.autograd.set_detect_anomaly(True)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

def solve_mpc(x0: torch.Tensor, P: torch.Tensor, B: torch.Tensor, obs: Observable, cost: Callable, h: int, umin=-1., umax=1.):
	'''
	h: horizon
	'''
	u = torch.full((1, h), 0).unsqueeze(2) 
	u = torch.nn.Parameter(u)
	opt = torch.optim.SGD([u], lr=0.1, momentum=0.9)

	loss, prev_loss = torch.Tensor([float('inf')]), torch.Tensor([0.])

	try:
		while torch.abs(loss - prev_loss).item() > 1e-6:
			x_pred = obs.extrapolate(P, x0.unsqueeze(1), h, B=B, u=u, build_graph=True, unlift_every=False)
			prev_loss = loss
			loss = cost(u, x_pred)
			print(loss.item())
			opt.zero_grad()
			loss.backward()
			opt.step()
			u.data = u.data.clamp(umin, umax)
	except KeyboardInterrupt:
		pass

	print('Final loss:', loss.item())

	return u.data

# Try it
data = hkl.load('saved/duffing_controlled_nominal.hkl')
P, B = torch.from_numpy(data['P']).float(), torch.from_numpy(data['B']).float()

x0 = torch.Tensor([-1.0, 0.0])
xR = 0.
cost = lambda u, x: ((x[0] - xR)**2).sum()
h = 50

u_opt = solve_mpc(x0, P, B, obs, cost, h)
plt.plot(u_opt[0])
plt.show()