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
p, d, k = 3, 2, 8
obs = PolynomialObservable(p, d, k)

# Initialize kernel
m, T = 2, 8
K = PFKernel(device, k, m, T, use_sqrt=False)

# Init data
mu = 2.0
X, Y = vdp.dataset(mu, skip=300)
P0 = edmd(X, Y, obs)
P0 = P0.to(device)
# P0 /= torch.norm(P0, 2)
print('Op valid:', PFKernel.validate(P0))

# HMC

logp = hmc.mk_log_prob(P0, K, rate=100) # increase rate to tighten uncertainty radius
samples, ratio = hmc.sample(10, logp, P0, step_size=.005)

print('Acceptance ratio: ', ratio)

# Visualize perturbations

x_0 = X[:,0]
t = 1000
# t = X.shape[1]

# Z0 = obs.preimage(torch.mm(P0, obs(X))).cpu()
Z0 = extrapolate(P0, x_0, obs, t)
plt.figure()
plt.title('Nominal prediction')
plt.plot(Z0[0], Z0[1])

for _ in range(3):
	Pn = random.choice(samples)
	# Zn = obs.preimage(torch.mm(Pn, obs(X))).cpu()
	Zn = extrapolate(Pn, x_0, obs, t)
	plt.figure()
	plt.title('Perturbed prediction (Kernel distance)')
	plt.plot(Zn[0], Zn[1])

for _ in range(3):
	sigma = 0.1
	eps = torch.distributions.Normal(torch.zeros_like(P0), torch.full(P0.shape, sigma)).sample()
	Pn = P0 + eps
	# Zn = obs.preimage(torch.mm(Pn, obs(X))).cpu()
	Zn = extrapolate(Pn, x_0, obs, t)
	plt.figure()
	plt.title('Perturbed prediction (Fro distance)')
	plt.plot(Zn[0], Zn[1])

plt.show()
