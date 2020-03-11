import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from operators import *
from utils import *
import hmc as hmc
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

set_seed(9001)

# Init features
p, d, k = 4, 2, 8
obs = PolynomialObservable(p, d, k)
# tau = 10
# obs = DelayObservable(tau) # Delay observable is not predictive, causes NaN

# Init data
mu = 3.0
X, Y = vdp.dataset(mu, n=2000, b=20)
X, Y = X.to(device), Y.to(device)
PsiX, PsiY = obs(X), obs(Y)

# Initialize kernel
d, m, T = PsiX.shape[0], 2, 18
K = PFKernel(device, d, m, T, use_sqrt=False)

# Nominal operator
P0 = dmd(PsiX, PsiY)
P0 = P0.to(device)
assert not torch.isnan(P0).any().item()

# HMC

baseline = False
eig_max = spectral_radius(P0)
s_max, s_min = eig_max, eig_max - 1e-3
dist_func = (lambda x, y: torch.norm(x - y)) if baseline else (lambda x, y: K(x, y, normalize=True)) 

alpha, beta = 1, 20
pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(device), torch.Tensor([beta]).to(device))

def potential(params: tuple):
	(P,) = params
	d_k = dist_func(P0, P).clamp(1e-6, 1-1e-6)
	print(d_k.item())
	u = -pdf.log_prob(d_k)
	return u

def boundary(params: tuple, momentum: tuple, step: float, resolution=10):
	(P,) = params
	(M,) = momentum
	s = spectral_radius(P + step*M)
	if s > s_max:
		micro_step = step/resolution
		P_cand = P.detach()
		while spectral_radius(P_cand) <= s_max:
			P_cand += micro_step*M
		P_cand = P_cand.requires_grad_(True)
		# Reflect along plane orthogonal to spectral gradient
		dS = -torch.autograd.grad(spectral_radius(P_cand), P_cand)[0]
		M_para = torch.trace(torch.mm(M.t(), dS)) * dS / torch.trace(torch.mm(dS.t(), dS))
		M_refl = M - 2*M_para
		return ((P_cand,), (M_refl,))
	elif s < s_min:
		micro_step = step/resolution
		P_cand = P.detach()
		while spectral_radius(P_cand) >= s_min:
			P_cand += micro_step*M
		P_cand = P_cand.requires_grad_(True)
		# Reflect along plane orthogonal to spectral gradient
		dS = torch.autograd.grad(spectral_radius(P_cand), P_cand)[0]
		M_para = torch.trace(torch.mm(M.t(), dS)) * dS / torch.trace(torch.mm(dS.t(), dS))
		M_refl = M - 2*M_para
		return ((P_cand,), (M_refl,))
	return None

samples, ratio = hmc.sample(10, (P0,), potential, boundary, n_leapfrog=25, step_size=.0001)
samples = [P.detach() for (P,) in samples]

print('Acceptance ratio: ', ratio)
print('Nominal spectral norm:', eig_max.item())
print('Perturbed spectral norms:', [spectral_radius(P).item() for P in samples])

# Save samples
name = 'perturbed_baseline' if baseline else 'perturbed_pf' 
torch.save(torch.stack(samples), f'tensors/{name}_vdp.pt')

# Visualize perturbations

t = 2000
# t = X.shape[1]

# plt.figure()
# plt.title('Nominal prediction')
# Z0 = obs.preimage(torch.mm(P0, obs(X))).cpu()
# plt.plot(Z0[0], Z0[1])

plt.figure()
plt.xlim(left=-6.0, right=6.0)
plt.ylim(bottom=-6.0, top=6.0)
plt.title(f'Perturbations of Van der Pol ({"baseline" if baseline else "kernel"})')
Z0 = obs.extrapolate(P0, X, t).cpu()
plt.plot(Z0[0], Z0[1], label='Nominal')

for Pn in samples:
	# Zn = obs.preimage(torch.mm(Pn, obs(X))).cpu()
	Zn = obs.extrapolate(Pn, X, t).cpu()
	plt.plot(Zn[0], Zn[1], color='orange')

plt.legend()

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

