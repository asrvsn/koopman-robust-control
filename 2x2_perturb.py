import torch
import numpy as np
import hickle as hkl

from sampler.utils import *
from sampler.kernel import *
from sampler.dynamics import *
from systems.lti2x2 import semistable_systems

set_seed(9001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.autograd.set_detect_anomaly(True)

baseline = True

if baseline:
	dist_func = euclidean_matrix_kernel
else:
	K = PFKernel(device, 2, 2, 80)
	dist_func = lambda x, y: K(x, y, normalize=True) 

n_show = 3
n_samples = 1000
n_split = 100
beta = 5
results = {}
step = 1e-4

for i, (name, A) in enumerate(semistable_systems.items()):

	nominal = torch.from_numpy(diff_to_transferop(A)).float()
	print(name, spectral_radius(nominal).item())
	
	samples = perturb(n_samples, nominal, beta, r_div=(1e-2, 1e-2), r_step=3e-5, dist_func=dist_func, n_split=n_split, hmc_step=step)
	sampledist = [dist_func(nominal, s).item() for s in samples]
	samples = [transferop_to_diff(s.numpy()) for s in samples]

	results[name] = {
		'nominal': A,
		'posterior': sampledist,
		'samples': samples,
	}

print('Saving..')
hkl.dump(results, f'saved/2x2{"_baseline" if baseline else ""}.hkl')
