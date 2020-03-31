import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np

from sampler.utils import *

method1, method2 = 'baseline', 'discounted_kernel'
results1 = hkl.load(f'saved/2x2_{method1}.hkl')
results2 = hkl.load(f'saved/2x2_{method2}.hkl')

plots = {
	# 'spiral_sink', 
	'spiral_sink2': 'Spiral sink', 
	'spiral_source': 'Spiral source', 
	'nodal_sink ': 'Nodal sink', 
	# 'nodal_source', # too explosive 
	'center1': 'Center', 
	'center2': 'Center', 
	'saddle1': 'Saddle', 
	# 'saddle2', 
	# 'saddle3', 
	# 'defective_source', # distribution not converged
	# 'defective_sink', 
}

fig, axs = plt.subplots(3, len(plots))

for i, (key, name) in enumerate(plots.items()):

	# Original phase portrait
	A = results1[key]['nominal']
	plot_flow_field(axs[0,i], lambda x: A@x, (-4,4), (-4,4))
	axs[0,i].set_title(name)
	axs[0,i].axis('off')

	# # Posterior distribution
	# posterior = results1[key]['posterior']
	# axs[1,i].hist(posterior, density=True, bins=max(1, int(len(posterior)/4))) 
	# axs[1,i].axes.get_yaxis().set_visible(False)

	# Trace-determinant plot
	samples = results1[key]['samples']
	tr = [np.trace(S) for S in samples]
	det = [np.linalg.det(S) for S in samples]
	axs[1,i].scatter(tr, det, color='blue', marker='+', alpha=0.5)
	plot_trace_determinant(axs[1,i], A, (-4,4), (-4,4))
	axs[1,i].axes.get_yaxis().set_visible(False)
	# axs[2,i].axes.get_xaxis().set_visible(False)

	# # Spectral radii
	# radii = [spectral_radius(torch.from_numpy(np.real(diff_to_transferop(s)))).item() for s in samples]
	# radius = spectral_radius(torch.from_numpy(diff_to_transferop(A))).item()
	# axs[3,i].scatter(radii, np.zeros(len(radii)), marker='+', color='blue')
	# axs[3,i].scatter([radius], [0], marker='+', color='orange')
	# axs[3,i].axes.get_yaxis().set_visible(False)
	# axs[3,i].set_aspect(0.2)

	# # Posterior distribution
	# posterior = results2[key]['posterior']
	# axs[3,i].hist(posterior, density=True, bins=max(1, int(len(posterior)/4))) 
	# axs[3,i].axes.get_yaxis().set_visible(False)

	# Trace-determinant plot
	samples = results2[key]['samples']
	tr = [np.trace(S) for S in samples]
	det = [np.linalg.det(S) for S in samples]
	axs[2,i].scatter(tr, det, color='blue', marker='+', alpha=0.5)
	plot_trace_determinant(axs[2,i], A, (-4,4), (-4,4))
	axs[2,i].axes.get_yaxis().set_visible(False)
	# axs[4,i].axes.get_xaxis().set_visible(False)

	# # Spectral radii
	# radii = [spectral_radius(torch.from_numpy(np.real(diff_to_transferop(s)))).item() for s in samples]
	# radius = spectral_radius(torch.from_numpy(diff_to_transferop(A))).item()
	# axs[6,i].scatter(radii, np.zeros(len(radii)), marker='+', color='blue')
	# axs[6,i].scatter([radius], [0], marker='+', color='orange')
	# axs[6,i].axes.get_yaxis().set_visible(False)
	# axs[6,i].set_aspect(0.2)

# axs[0,0].set_ylabel('phase portrait')
# axs[1,0].set_ylabel('posterior\n(baseline)')
# axs[2,0].set_ylabel('trace-determinant\n(baseline)')
# axs[3,0].set_ylabel('spectral radii\n(baseline)')
# axs[3,0].set_ylabel('posterior\n(P-F)')
# axs[4,0].set_ylabel('trace-determinant\n(P-F)')
# axs[6,0].set_ylabel('spectral radii\n(P-F)')

# fig.suptitle(f'Perturbations of 2x2 LTI semistable_systems ({method})')
# fig.tight_layout(pad=0.01)
plt.show()