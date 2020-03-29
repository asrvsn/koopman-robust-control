import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np

from sampler.utils import *

method = 'baseline'
results = hkl.load(f'saved/2x2_{method}.hkl')

plots = [
	'spiral_sink', 
	'spiral_sink2', 
	'spiral_source', 
	'nodal_sink ', 
	'center1', 
	'center2', 
	'saddle1', 
	'saddle2', 
	'saddle3', 
	'defective_source',
	'defective_sink', 
]

fig, axs = plt.subplots(4, len(plots))

for i, name in enumerate(plots):

	data = results[name]
	A = data['nominal']
	posterior = data['posterior']
	samples = data['samples']

	# Original phase portrait
	plot_flow_field(axs[0,i], lambda x: A@x, (-4,4), (-4,4))
	axs[0,i].set_title(name)

	# Perturbed phase portraits
	# axs[1,i].set_title('Perturbed')
	# for j, S in enumerate(random.choices(samples, k=n_show)):
	# 	plot_flow_field(axs[j+1,i], lambda x: S@x, (-4,4), (-4,4))

	# Posterior distribution
	axs[1,i].hist(posterior, density=True, bins=max(1, int(len(posterior)/4))) 

	# Trace-determinant plot
	tr = [np.trace(S) for S in samples]
	det = [np.linalg.det(S) for S in samples]
	axs[2,i].scatter(tr, det, color='blue', marker='+', alpha=0.5)
	plot_trace_determinant(axs[-2,i], A, (-4,4), (-4,4))

	# Spectral radii
	radii = [spectral_radius(torch.from_numpy(np.real(diff_to_transferop(s)))).item() for s in samples]
	radius = spectral_radius(torch.from_numpy(diff_to_transferop(A))).item()
	axs[3,i].scatter(radii, np.zeros(len(radii)), marker='+', color='blue')
	axs[3,i].scatter([radius], [0], marker='+', color='orange')

axs[0,0].set_ylabel('phase portrait')
axs[1,0].set_ylabel('posterior')
axs[2,0].set_ylabel('trace-determinant')
axs[3,0].set_ylabel('spectral radii')

# fig.suptitle(f'Perturbations of 2x2 LTI semistable_systems ({method})')
# fig.tight_layout(pad=0.01)
plt.show()