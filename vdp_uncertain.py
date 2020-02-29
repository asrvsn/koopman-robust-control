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
m, T = 2, 4
K = PFKernel(device, k, m, T)

# Init data
mu = 0
X, Y = vdp.dataset(mu)
P0 = edmd(X, Y, obs)
P0 = P0.to(device)

# HMC
logp = mk_log_prob(P0, K)
samples = sample(10, logp, P0, seed=9001)

plt.figure()
plt.title('Kernel distance vs. mu')
plt.yscale('log')
plt.plot(mu_rng, results[:, 0])
plt.figure()
plt.title('Frobenius distance vs. mu')
plt.plot(mu_rng, results[:, 1])

plt.show()
