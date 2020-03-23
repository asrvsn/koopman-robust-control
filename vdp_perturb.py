import torch
import hickle as hkl

from sampler.features import *
from sampler.kernel import *
from sampler.operators import *
from sampler.utils import *
from sampler.dynamics import *
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

# Nominal operator
nominal = dmd(PsiX, PsiY)
nominal = nominal.to(device)
assert not torch.isnan(nominal).any().item()

# Sample dynamics
baseline = False

if baseline:
	dist_func = euclidean_matrix_kernel
else:
	d, m, T = PsiX.shape[0], 2, 80
	K = PFKernel(device, d, m, T)
	dist_func = lambda x, y: K(x, y, normalize=True) 

beta = 5
step=1e-4
n_samples = 200
n_split = 20
burn = 5

samples = perturb(n_samples, nominal, beta, dist_func=dist_func, r_div=(1e-2, 1e-2), r_step=3e-5, n_split=n_split, hmc_step=step, hmc_burn=burn)
posterior = [dist_func(nominal, s).item() for s in samples]

results = {
	'nominal': nominal.numpy(),
	'posterior': posterior,
	'samples': [s.numpy() for s in samples],
	'step': step,
	'beta': beta,
}
print('Saving..')
hkl.dump(results, f'saved/vdp{"_baseline" if baseline else ""}.hkl')
