import torch
import matplotlib.pyplot as plt
import scipy.stats as stats

from features import *
from kernel import *
from operators import *
from utils import *
from sampler.dynamics import perturb
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.autograd.set_detect_anomaly(True)

set_seed(9001)

# Init features
p, d, k = 4, 2, 8
obs = PolynomialObservable(p, d, k)

# Init data
mu = 3.0
n_vdp = 6000
b_vdp = 40
skip_vdp = int(n_vdp * 8 / b_vdp) # start on limit cycle
X, Y = vdp.dataset(mu, n=n_vdp, b=b_vdp, skip=skip_vdp) 
X, Y = X.to(device), Y.to(device)
PsiX, PsiY = obs(X), obs(Y)

# Initialize kernel
d, m, T = PsiX.shape[0], 2, 80
K = PFKernel(device, d, m, T)

# Nominal operator
P0 = dmd(PsiX, PsiY)
P0 = P0.to(device)
assert not torch.isnan(P0).any().item()

# Sample dynamics

baseline = False
beta = 10
dist_func = euclidean_matrix_kernel if baseline else (lambda x, y: K(x, y, normalize=True)) 
hmc_step=1e-4
n_samples = 200

samples = perturb(
	n_samples, P0, dist_func, beta,  
	sp_div=(1e-1, 1e-1),
	hmc_step=hmc_step,
  hmc_random_step=True,
	hmc_leapfrog=25,
  hmc_burn=30,
  hmc_target_accept=0.75,
  NUTS=False,
)

# Save samples

name = 'perturbed_baseline' if baseline else 'perturbed_pf' 
torch.save(torch.stack(samples), f'tensors/{name}_vdp.pt')

# Visualize perturbations

t = int(n_vdp/2)

X = X.cpu()

plt.figure()

for Pn in samples:
	Zn = obs.extrapolate(Pn.cpu(), X, t)
	plt.plot(Zn[0], Zn[1], color='grey', alpha=0.3, label='Numeric perturbation')

# for mu in [2.5, 3.5, 4.0]:
#   X, _ = vdp.dataset(mu, n=n_vdp, b=b_vdp, skip=skip_vdp) # start on limit cycle
#   X = X[:t]
#   plt.plot(X[0], X[1], color='green', alpha=0.5, label='Analytic perturbation')

plt.xlim(left=-6.0, right=6.0)
plt.ylim(bottom=-6.0, top=6.0)
plt.title(f'beta={beta}, step={hmc_step}, T={T}, distance={"Frobenius" if baseline else "Kernel"}')
Z0 = obs.extrapolate(P0.cpu(), X, t)
plt.plot(Z0[0], Z0[1], label='Nominal')

plt.plot([X[0][0]], [X[1][0]], marker='o', color='blue')
deduped_legend()

# Sample distribution

plt.figure()
distances = [dist_func(P0, P) for P in samples]
plt.hist(distances, bins=int(n_samples/10), density=True)
d = np.linspace(0, 1, 100)
p = stats.beta.pdf(d, 1.0, beta, loc=0, scale=1)
plt.plot(d, p)
plt.title(f'Posterior distribution, beta={beta}')

# Closeup

# t = 300

# plt.figure()
# for Pn in samples:
# 	Zn = obs.extrapolate(Pn.cpu(), X, t+50)
# 	plt.plot(Zn[0], Zn[1], color='grey', alpha=0.3)

# plt.xlim(left=X[0][t], right=X[0][0] + 0.05)
# plt.ylim(bottom=X[1][t], top=X[1][0] + 0.05)
# plt.title(f'beta={beta}, step={hmc_step}, T={T}, distance={"Frobenius" if baseline else "Kernel"}')
# Z0 = obs.extrapolate(P0.cpu(), X, t+100)
# plt.plot(Z0[0], Z0[1], label='Nominal')

# plt.plot([X[0][0]], [X[1][0]], marker='o', color='blue')
# plt.legend()

plt.show()

