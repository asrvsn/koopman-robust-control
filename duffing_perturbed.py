import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from operators import *
from utils import *
from sampler.dynamics import perturb
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
n_init = 12
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

# Sample dynamics

baseline = False
beta = 50
dist_func = euclidean_matrix_kernel if baseline else (lambda x, y: K(x, y, normalize=True)) 
hmc_step = 1e-5
x_samples = 6
y_samples = 3
n_samples = x_samples*y_samples

samples = perturb(
	n_samples, P0, dist_func, beta,  
	sp_div=(1e-2, 1e-2),
	hmc_step=hmc_step,
	hmc_leapfrog=25,
)

# Save samples
name = 'perturbed_baseline' if baseline else 'perturbed_pf' 
torch.save(torch.stack(samples), f'tensors/{name}_duffing.pt')

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
plt.title('Nominal extrapolation of the Duffing Oscillator')
for x0 in x0s:
	for xdot0 in xdot0s:
		x = torch.Tensor([[x0], [xdot0]]).to(device)
		Z0 = obs.extrapolate(P0, x, t).cpu()
		label, color = get_label(Z0[:, -1])
		if label != 'Unstable':
			plt.plot(Z0[0], Z0[1], label=label, color=color)

deduped_legend()

fig, axs = plt.subplots(y_samples, x_samples)
fig.suptitle(f'Perturbed extrapolations of the Duffing oscillator')
r, c = 0, 0
for Pn in samples:
	ax = axs[r, c]
	xbound, ybound = 2.2, 2.2
	ax.set_xlim(left=-xbound, right=xbound)
	ax.set_ylim(bottom=-ybound, top=ybound)
	for x0 in x0s:
		for xdot0 in xdot0s:
			x = torch.Tensor([[x0], [xdot0]]).to(device)
			Zn = obs.extrapolate(Pn, x, t).cpu()
			label, color = get_label(Zn[:, -1])
			if label != 'Unstable':
				ax.plot(Zn[0], Zn[1], label=label, color=color)
	c += 1
	if c == x_samples:
		c = 0
		r += 1

deduped_legend()

plt.show()

