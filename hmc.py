import torch
import numpy as np

def hmc(logprob: Callable, X0: torch.Tensor, n_samples: int, n_steps=1, n_omit=0, decay=0.9, epsilon=0.2, window=1, seed=None):
	'''
	Torch-based Hamiltonian Monte Carlo sampling. Based on https://github.com/rmcgibbo/pyhmc/blob/master/pyhmc/hmc.py 
	'''
	assert n_samples >= 1 and n_steps >= 1 and n_omit >= 0 and n_decay <= 1
	assert X0.shape[0] == X0.shape[1]

	if seed is not None:
		np.random.seed(seed)
		torch.manual_seed(seed)

	dev = X0.device
	d = X0.shape[0]
	samples = Torch.Tensor((n_samples, d, d)) # returned samples 
	window_offset = 0

	X, P = X0.clone(), torch.randn((d, d))
	X_old, P_old = X0, P.clone()
	X_acc, P_acc = torch.zeros((d, d), device=dev), torch.zeros((d, d), device=dev)
	X_rej, P_rej = torch.zeros((d, d), device=dev), torch.zeros((d, d), device=dev)

	logp, grad = logprob(X0)
	E = -logp

	k = -n_omit # omitted samples (expected time to reach stationary dist.)
	while k < n_samples:
		E_old, H_old = E, E + 0.5*torch.trace(torch.mm(P.t(), P))

		if window > 1:
			window_offset = int(window * random.rand())

		have_rej = 0
		have_acc = 0
		n = window_offset
		direction = -1

		# TODO... https://github.com/rmcgibbo/pyhmc/blob/master/pyhmc/_hmc.pyx#L70