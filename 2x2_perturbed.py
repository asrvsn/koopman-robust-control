import torch
import numpy as np
import matplotlib.pyplot as plt

import settings as settings
if settings.GCP:
	torch.multiprocessing.set_sharing_strategy('file_system')
	import matplotlib
	matplotlib.use('tkagg')

from sampler.utils import *
from sampler.kernel import *
from sampler.dynamics import *
from systems.lti2x2 import semistable_systems

set_seed(9001)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.autograd.set_detect_anomaly(True)

K = PFKernel(device, 2, 2, 80)
dist_func = lambda x, y: K(x, y, normalize=True) 

n_show = 3
n_samples = 1000
n_split = 100
beta = 5
fig, axs = plt.subplots(3 + n_show, len(semistable_systems))

for i, (name, A) in enumerate(semistable_systems.items()):

	expA = torch.from_numpy(diff_to_transferop(A)).float()
	print(name, spectral_radius(expA).item())
	
	samples = perturb(n_samples, expA, beta, r_div=(1e-2, 1e-2), r_step=3e-5, dist_func=dist_func, n_split=n_split)
	sampledist = [dist_func(expA, s).item() for s in samples]
	samples = [transferop_to_diff(s.numpy()) for s in samples]

	# Original phase portrait
	plot_flow_field(axs[0,i], lambda x: A@x, (-4,4), (-4,4))
	axs[0,i].set_title(f'{name} - original')

	# Posterior distribution
	axs[1,i].hist(sampledist, density=True, bins=max(1, int(n_samples/4))) 
	axs[1,i].set_title('Posterior distribution')

	# Trace-determinant plot
	plot_trace_determinant(axs[2,i], A, (-4,4), (-4,4))
	tr = [np.trace(S) for S in samples]
	det = [np.linalg.det(S) for S in samples]
	axs[2,i].scatter(tr, det, color='blue', alpha=0.5)
	axs[2,i].set_title('Trace-determinant plane')

	# Perturbed phase portraits
	for j, S in enumerate(random.choices(samples, k=n_show)):
		plot_flow_field(axs[j+3,i], lambda x: S@x, (-4,4), (-4,4))

fig.suptitle('Perturbations of 2x2 LTI semistable_systems')

plt.show()