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
# P0 /= torch.norm(P0, 2)
assert not torch.isnan(P0).any().item()

# HMC

baseline = True
eig_max = torch.norm(P0, p=2) 
s_max, s_min = eig_max + 1e-3, eig_max - 1e-3
dist_func = (lambda x, y: torch.norm(x - y)) if baseline else (lambda x, y: K(x, y, normalize=True)) 

alpha, beta = 1, 8
pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(device), torch.Tensor([beta]).to(device))

def potential(params: tuple):
	(P,) = params
	d_k = dist_func(P0, P).clamp(1e-6, 1-1e-6)
	print(d_k.item())
	rate = 10.0
	u = -pdf.log_prob(d_k)
	return u

def boundary(params: tuple, momentum: tuple, step: float, resolution=10):
	(P,) = params
	(M,) = momentum
	s = torch.norm(P + step*M, p=2)
	if s > s_max:
		micro_step = step/resolution
		P_cand = P.detach()
		while torch.norm(P_cand, p=2) <= s_max:
			P_cand += micro_step*M
		P_cand = P_cand.requires_grad_(True)
		# Reflect along plane orthogonal to spectral gradient
		dS = -torch.autograd.grad(torch.norm(P_cand, p=2), P_cand)[0]
		M_para = torch.trace(torch.mm(M.t(), dS)) * dS / torch.trace(torch.mm(dS.t(), dS))
		M_refl = M - 2*M_para
		return ((P_cand,), (M_refl,))
	elif s < s_min:
		micro_step = step/resolution
		P_cand = P.detach()
		while torch.norm(P_cand, p=2) >= s_min:
			P_cand += micro_step*M
		P_cand = P_cand.requires_grad_(True)
		# Reflect along plane orthogonal to spectral gradient
		dS = torch.autograd.grad(torch.norm(P_cand, p=2), P_cand)[0]
		M_para = torch.trace(torch.mm(M.t(), dS)) * dS / torch.trace(torch.mm(dS.t(), dS))
		M_refl = M - 2*M_para
		return ((P_cand,), (M_refl,))
	return None

samples, ratio = hmc.sample(10, (P0,), potential, boundary, n_leapfrog=50, step_size=.0001)
samples = [P.detach() for (P,) in samples]

print('Acceptance ratio: ', ratio)
print('Nominal spectral norm:', eig_max.item())
print('Perturbed spectral norms:', [torch.norm(P, p=2).item() for P in samples])

# Save samples
name = 'perturbed_baseline' if baseline else 'perturbed_pf' 
torch.save(torch.stack(samples), f'{name}.pt')

# Visualize perturbations

t = 2000
# t = X.shape[1]

# plt.figure()
# plt.title('Nominal prediction')
# Z0 = obs.preimage(torch.mm(P0, obs(X))).cpu()
# plt.plot(Z0[0], Z0[1])

plt.figure()
plt.xlim(left=-4.0, right=4.0)
plt.ylim(bottom=-4.0, top=4.0)
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

