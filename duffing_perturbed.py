import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from operators import *
from utils import *
import hmc as hmc
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

set_seed(9001)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Init data 
t_max = 400
n_per = 16000
n_init = 11
x0s = np.linspace(-2.0, 2.0, n_init)
xdot0s = np.linspace(-2.0, 2.0, n_init)
X, Y = [], []
for x0 in x0s:
	for xdot0 in xdot0s:
		# Unforced duffing equation
		Xi, Yi = duffing.dataset(t_max, n_per, gamma=0.0, x0=x0, xdot0=xdot0)
		X.append(Xi)
		Y.append(Yi)
X, Y = torch.cat(tuple(X), axis=1), torch.cat(tuple(Y), axis=1)

X, Y = X.to(device), Y.to(device)
PsiX, PsiY = obs(X), obs(Y)

# Initialize kernel
d, m, T = PsiX.shape[0], 2, 18
K = PFKernel(device, d, m, T)

# Nominal operator
P0 = dmd(PsiX, PsiY)
P0 = P0.to(device)
assert not torch.isnan(P0).any().item()

# HMC

baseline = False
eig_max = spectral_radius(P0) 
s_max, s_min = eig_max + 1e-3, eig_max - 1e-3
dist_func = euclidean_matrix_kernel if baseline else (lambda x, y: K(x, y, normalize=True)) 

alpha, beta = 1, 80
pdf = torch.distributions.beta.Beta(torch.Tensor([alpha]).to(device), torch.Tensor([beta]).to(device))

def potential(params: tuple):
	(P,) = params
	d_k = dist_func(P0, P).clamp(1e-6, 1-1e-6)
	print(d_k.item())
	u = pdf.log_prob(d_k)
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

n_samples = 6
samples, ratio = hmc.sample(n_samples, (P0,), potential, boundary, n_burn=20, n_leapfrog=25, step_size=.00001)
samples = [P for (P,) in samples]

print('Acceptance ratio: ', ratio)
print('Nominal spectral radius:', eig_max.item())
print('Perturbed spectral radii:', [spectral_radius(P).item() for P in samples])

# Save samples
# name = 'perturbed_baseline' if baseline else 'perturbed_pf' 
# torch.save(torch.stack(samples), f'tensors/{name}_duffing.pt')

# Visualize perturbations

t = 800

def get_label(z):
	if z[0] > -1.5 and z[0] < -0.5 and z[1] > -0.5 and z[1] < 0.5:
		return 'Left', 'red'
	elif z[0] > 0.5 and z[0] < 1.5 and z[1] > -0.5 and z[1] < 0.5:
		return 'Right', 'blue'
	else:
		return 'Unstable', 'grey'

plt.figure()
xbound, ybound = 2.2, 2.2
plt.xlim(left=-xbound, right=xbound)
plt.ylim(bottom=-ybound, top=ybound)
plt.title('Nominal extrapolation of Duffing Oscillator')
for x0 in x0s:
	for xdot0 in xdot0s:
		x = torch.Tensor([[x0], [xdot0]]).to(device)
		Z0 = obs.extrapolate(P0, x, t).cpu()
		label, color = get_label(Z0[:, -1])
		if label != 'Unstable':
			plt.plot(Z0[0], Z0[1], label=label, color=color)

deduped_legend()

for Pn in samples:
	plt.figure()
	xbound, ybound = 2.2, 2.2
	plt.xlim(left=-xbound, right=xbound)
	plt.ylim(bottom=-ybound, top=ybound)
	plt.title(f'Perturbed extrapolations of Duffing oscillator ({"Frobenius distance" if baseline else "P-F distance"})')
	for x0 in x0s:
		for xdot0 in xdot0s:
			x = torch.Tensor([[x0], [xdot0]]).to(device)
			Zn = obs.extrapolate(Pn, x, t).cpu()
			label, color = get_label(Zn[:, -1])
			if label != 'Unstable':
				plt.plot(Zn[0], Zn[1], label=label, color=color)

	deduped_legend()

plt.show()

