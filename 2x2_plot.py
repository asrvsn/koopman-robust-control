import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np

from sampler.utils import *

results = hkl.load('saved/2x2.hkl')

n_show = 3
fig, axs = plt.subplots(3 + n_show, len(results))

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
	axs[-2,i].hist(posterior, density=True, bins=max(1, int(len(posterior)/4))) 
	# axs[-2,i].set_title('Posterior distribution')

	# Trace-determinant plot
	plot_trace_determinant(axs[-1,i], A, (-4,4), (-4,4))
	tr = [np.trace(S) for S in samples]
	det = [np.linalg.det(S) for S in samples]
	axs[-1,i].scatter(tr, det, color='blue', marker='+', alpha=0.5)
	# axs[-1,i].set_title('Trace-determinant plane')

fig.suptitle('Perturbations of 2x2 LTI semistable_systems')
# fig.tight_layout(pad=0.01)
plt.show()