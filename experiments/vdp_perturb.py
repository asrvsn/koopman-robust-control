import torch
import hickle as hkl
import multiprocessing

import sampler.reflections as reflections
from sampler.features import *
from sampler.kernel import *
from sampler.operators import *
from sampler.utils import *
from sampler.ugen import *
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
X, Y = vdp.dataset(mu, n=n_vdp, b=b_vdp) 
X, Y = X.to(device), Y.to(device)
PsiX, PsiY = obs(X), obs(Y)

# Nominal operator
nominal = dmd(PsiX, PsiY)
nominal = nominal.to(device)
assert not torch.isnan(nominal).any().item()

# Sample dynamics

method = 'baseline'
# method = 'kernel'
# method = 'constrained_kernel'
# method = 'discounted_kernel'

beta = 20
step = 1e-5
leapfrog = 200
n_samples = 1000
n_ics = 10
ic_step = 1e-5
T = 80
L = 0.1

if __name__ == '__main__':

	try:
	    multiprocessing.set_start_method('spawn')
	except RuntimeError:
	    pass
	multiprocessing.freeze_support()

	if method == 'baseline':
		samples, posterior = perturb(n_samples, nominal, beta, method='euclidean', n_ics=n_ics, hmc_step=step, hmc_leapfrog=leapfrog, ic_step=ic_step)
	elif method == 'kernel':
		samples, posterior = perturb(n_samples, nominal, beta, method='kernel', n_ics=n_ics, hmc_step=step, hmc_leapfrog=leapfrog, ic_step=ic_step, kernel_T=T)
	elif method == 'constrained_kernel':
		samples, posterior = perturb(n_samples, nominal, beta, method='kernel', n_ics=n_ics, hmc_step=step, hmc_leapfrog=leapfrog, ic_step=ic_step, kernel_T=T, use_spectral_constraint=True)
	elif method == 'discounted_kernel':
		samples, posterior = perturb(n_samples, nominal, beta, method='kernel', n_ics=n_ics, hmc_step=step, hmc_leapfrog=leapfrog, ic_step=ic_step, kernel_T=T, kernel_L=L)

	# print('Saving trajectories...')
	# n_trajectories = 12
	# n_ics = 12
	# t = 3000
	# trajectories = [sample_2d_dynamics(P, obs, t, (-6,6), (-6,6), n_ics, n_ics) for P in random.choices(samples, k=n_trajectories)]

	results = {
		'method': method,
		'step': step,
		'beta': beta,
		'nominal': nominal.cpu().numpy(),
		'posterior': posterior,
		'samples': [s.cpu().numpy() for s in samples],
		# 'trajectories': trajectories,
	}
	print('Saving..')
	hkl.dump(results, f'saved/vdp_{method}.hkl')
