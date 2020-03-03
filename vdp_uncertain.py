import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from operators import *
from utils import set_seed
import hmc as hmc
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

set_seed(9001)

# Init features
p, d, k = 4, 2, 5
obs = PolynomialObservable(p, d, k)

# Initialize kernel
m, T = 2, 12
K = PFKernel(device, k, m, T, use_sqrt=False)

# Init data
mu = 2.0
X, Y = vdp.dataset(mu, skip=200)
X, Y = X.to(device), Y.to(device)
P0 = edmd(X, Y, obs)
P0 = P0.to(device)
# P0 /= torch.norm(P0, 2)
print('Op valid:', PFKernel.validate(P0))

# HMC

pf_thresh = 1e-5

potential = hmc.mk_potential(P0, K, rate=100, eps=pf_thresh) # increase rate to tighten uncertainty radius
samples, ratio = hmc.sample(10, potential, P0, step_size=.0001, pf_thresh=pf_thresh)

print('Acceptance ratio: ', ratio)
(eig, _) = torch.eig(P0)
print('Nominal spectral norm:', eig.max().item())
print('Nominal valid:', PFKernel.validate(P0))
print('Perturbed valid:', [PFKernel.validate(P, eps=pf_thresh) for P in samples])

# Visualize perturbations

x_0 = X[:,0]
t = 5000
# t = X.shape[1]

plt.figure()
plt.title('Nominal prediction')
Z0 = obs.preimage(torch.mm(P0, obs(X))).cpu()
plt.plot(Z0[0], Z0[1])

plt.figure()
plt.title('Nominal extrapolation')
Z0 = extrapolate(P0, x_0, obs, t).cpu()
plt.plot(Z0[0], Z0[1])

for Pn in samples:
	# Zn = obs.preimage(torch.mm(Pn, obs(X))).cpu()
	Zn = extrapolate(Pn, x_0, obs, t).cpu()
	plt.figure()
	plt.title('Perturbed extrapolation (Kernel distance)')
	plt.plot(Zn[0], Zn[1])

# for _ in range(3):
# 	sigma = 0.1
# 	eps = torch.distributions.Normal(torch.zeros_like(P0, device=device), torch.full(P0.shape, sigma, device=device)).sample()
# 	Pn = P0 + eps
# 	# Zn = obs.preimage(torch.mm(Pn, obs(X))).cpu()
# 	Zn = extrapolate(Pn, x_0, obs, t).cpu()
# 	plt.figure()
# 	plt.title('Perturbed extrapolation (Fro distance)')
# 	plt.plot(Zn[0], Zn[1])

plt.show()
