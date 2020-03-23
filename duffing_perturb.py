import torch
import matplotlib.pyplot as plt
import hickle as hkl

from sampler.features import *
from sampler.kernel import *
from sampler.operators import *
from sampler.utils import *
from sampler.dynamics import *
import systems.duffing as duffing

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.autograd.set_detect_anomaly(True)

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

# Nominal operator
nominal = dmd(PsiX, PsiY)
nominal = nominal.to(device)
assert not torch.isnan(nominal).any().item()

# Sample dynamics
baseline = True

if baseline:
	dist_func = euclidean_matrix_kernel
else:
	d, m, T = PsiX.shape[0], 2, 20
	K = PFKernel(device, d, m, T)
	dist_func = lambda x, y: K(x, y, normalize=True) 


beta = 5
step = 1e-4
x_samples = 6
y_samples = 3
n_samples = 100
n_split = 20

samples = perturb(n_samples, nominal, beta, dist_func=dist_func, r_div=(1e-2, 1e-2), r_step=3e-5, n_split=n_split, hmc_step=step)
sampledist = [dist_func(nominal, s).item() for s in samples]

results = {
	'nominal': nominal.numpy(),
	'posterior': sampledist,
	'samples': [s.numpy() for s in samples],
}
print('Saving..')
hkl.dump(results, f'saved/duffing{"_baseline" if baseline else ""}.hkl')
