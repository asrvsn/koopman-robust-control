import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from operators import *
from hmc import *
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

# Init features
p, d, k = 10, 2, 20
obs = PolynomialObservable(p, d, k, seed=9001)

# Initialize kernel
m, T = 2, 8
K = PFKernel(device, k, m, T)

# Init data
mu = 0
X, Y = vdp.dataset(mu)
P0 = edmd(X, Y, obs)
P0 = P0.to(device)

# HMC
def mk_log_prob(P0: torch.Tensor, K: PFKernel, rate=100.0):
	rate = torch.Tensor([rate]).to(P0.device)
	dist = torch.distributions.exponential.Exponential(rate)
	def log_prob(P1: torch.Tensor):
		d_pf = K(P0, P1, normalize=True)
		return dist.log_prob(d_pf)
	return log_prob

logp = mk_log_prob(P0, K)

mu = 10.0
X, Y = vdp.dataset(mu)
P1 = edmd(X, Y, obs)
P1 = P1.to(device).requires_grad_()

grad = torch.autograd.grad(logp(P1), P1)
print(grad)


# logp = mk_log_prob(P0, K)
# samples = sample(10, logp, P0, seed=9001)

# Z0 = obs.preimage(torch.mm(P0, obs(X))).cpu()
# Zn = obs.preimage(torch.mm(samples[-1], obs(X))).cpu()

# plt.figure()
# plt.title('Nominal prediction')
# plt.plot(Z0[0], Z0[1])
# plt.figure()
# plt.title('Perturbed prediction')
# plt.plot(Zn[0], Zn[1])

# plt.show()
