import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np

from sampler.utils import *

method = 'constrained_kernel'
results = hkl.load(f'saved/2x2_{method}.hkl')

n_show = 3
fig, axs = plt.subplots(4 + n_show, len(results))

for i, (name, data) in enumerate(results.items()):

	A = data['nominal']
	posterior = data['posterior']
	samples = data['samples']

	# Original phase portrait
	plot_flow_field(axs[0,i], lambda x: A@x, (-4,4), (-4,4))
	axs[0,i].set_title(f'{name} - nominal')

	# Perturbed phase portraits
	axs[1,i].set_title('Perturbed')
	for j, S in enumerate(random.choices(samples, k=n_show)):
		plot_flow_field(axs[j+1,i], lambda x: S@x, (-4,4), (-4,4))

	# Posterior distribution
	axs[-3,i].hist(posterior, density=True, bins=max(1, int(len(posterior)/4))) 
	# axs[-2,i].set_title('Posterior distribution')

	# Trace-determinant plot
	tr = [np.trace(S) for S in samples]
	det = [np.linalg.det(S) for S in samples]
	axs[-2,i].scatter(tr, det, color='blue', marker='+', alpha=0.5)
	plot_trace_determinant(axs[-2,i], A, (-4,4), (-4,4))
	# axs[-1,i].set_title('Trace-determinant plane')

	# Spectral radii
	radii = [spectral_radius(torch.from_numpy(np.real(diff_to_transferop(s)))).item() for s in samples]
	radius = spectral_radius(torch.from_numpy(diff_to_transferop(A))).item()
	axs[-1,i].scatter(radii, np.zeros(len(radii)), marker='+', color='blue')
	axs[-1,i].scatter([radius], [0], marker='+', color='orange')

fig.suptitle(f'Perturbations of 2x2 LTI semistable_systems ({method})')
# fig.tight_layout(pad=0.01)
plt.show()