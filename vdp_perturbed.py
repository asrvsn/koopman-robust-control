import torch
import matplotlib.pyplot as plt

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
X, Y = vdp.dataset(mu, n=6000, b=40, skip=1312) # start on limit cycle
X, Y = X.to(device), Y.to(device)
PsiX, PsiY = obs(X), obs(Y)

# Initialize kernel
d, m, T = PsiX.shape[0], 2, 60
K = PFKernel(device, d, m, T)

# Nominal operator
P0 = dmd(PsiX, PsiY)
P0 = P0.to(device)
assert not torch.isnan(P0).any().item()

# Sample dynamics

baseline = False
beta = 200
dist_func = euclidean_matrix_kernel if baseline else (lambda x, y: K(x, y, normalize=True)) 
hmc_step=1e-4

samples = perturb(
	50, P0, dist_func, beta,  
	sp_div=(1e-1, 1e-1),
	hmc_step=hmc_step,
	hmc_leapfrog=25,
)

# Save samples

name = 'perturbed_baseline' if baseline else 'perturbed_pf' 
torch.save(torch.stack(samples), f'tensors/{name}_vdp.pt')

# Visualize perturbations

t = 3000

X = X.cpu()

plt.figure()
for Pn in samples:
	Zn = obs.extrapolate(Pn.cpu(), X, t)
	plt.plot(Zn[0], Zn[1], color='grey', alpha=0.3)

plt.xlim(left=-6.0, right=6.0)
plt.ylim(bottom=-6.0, top=6.0)
plt.title(f'beta={beta}, step={hmc_step}, T={T}, distance={"Frobenius" if baseline else "Kernel"}')
Z0 = obs.extrapolate(P0.cpu(), X, t)
plt.plot(Z0[0], Z0[1], label='Nominal')

plt.plot([X[0][0]], [X[1][0]], marker='o', color='blue')
plt.legend()

t = 300

plt.figure()
for Pn in samples:
	Zn = obs.extrapolate(Pn.cpu(), X, t+50)
	plt.plot(Zn[0], Zn[1], color='grey', alpha=0.3)

plt.xlim(left=X[0][t], right=X[0][0] + 0.05)
plt.ylim(bottom=X[1][t], top=X[1][0] + 0.05)
plt.title(f'beta={beta}, step={hmc_step}, T={T}, distance={"Frobenius" if baseline else "Kernel"}')
Z0 = obs.extrapolate(P0.cpu(), X, t+100)
plt.plot(Z0[0], Z0[1], label='Nominal')

plt.plot([X[0][0]], [X[1][0]], marker='o', color='blue')
plt.legend()

plt.show()

