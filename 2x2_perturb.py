import torch
import numpy as np
import hickle as hkl

import sampler.reflections as reflections
from sampler.utils import *
from sampler.kernel import *
from sampler.dynamics import *
from systems.lti2x2 import semistable_systems

set_seed(9001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.autograd.set_detect_anomaly(True)

# method = 'baseline'
# method = 'kernel'
method = 'constrained_kernel'

n_show = 3
n_samples = 1000
n_split = 50
beta = 5
results = {}
step = 1e-4

for i, (name, A) in enumerate(semistable_systems.items()):

	nominal = torch.from_numpy(diff_to_transferop(A)).float()
	
	if method == 'baseline':
		samples, posterior = perturb(n_samples, nominal, beta, dist_func=euclidean_matrix_kernel, boundary=reflections.nil_boundary, n_split=n_split, hmc_step=step)
	elif method == 'kernel':
		T = 80
		samples, posterior = perturb(n_samples, nominal, beta, boundary=reflections.nil_boundary, n_split=n_split, hmc_step=step, kernel_T=T)
	elif method == 'constrained_kernel':
		# print(name, spectral_radius(nominal).item())
		r_div=(1e-2, 1e-2)
		r_step = 3e-5
		T = 80
		samples, posterior = perturb(n_samples, nominal, beta, r_div=r_div, r_step=r_step, n_split=n_split, hmc_step=step, kernel_T=T)

	samples = [transferop_to_diff(s.numpy()) for s in samples]

	results[name] = {
		'nominal': A,
		'posterior': posterior,
		'samples': samples,
	}

print('Saving..')
hkl.dump(results, f'saved/2x2_{method}.hkl')
