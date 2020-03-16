from typing import Callable
from collections import OrderedDict
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def zip_with(X: tuple, Y: tuple, f: Callable):
	return tuple(f(x,y) for (x,y) in zip(X, Y))

def spectral_radius(A: torch.Tensor, eps=1e-6):
	v = torch.randn((A.shape[0], 1), device=A.device)
	v_new = v.clone()
	ev = v.t()@A@v
	ev_new = ev.clone()
	while (A@v_new - ev_new*v_new).norm() > eps:
		v = v_new
		ev = ev_new
		v_new = A@v
		v_new /= v_new.norm()
		ev_new = v_new.t()@A@v_new
	return ev_new

def deduped_legend():
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())

def euclidean_matrix_kernel(A: torch.Tensor, B: torch.Tensor):
	return torch.sqrt((1 - torch.trace(torch.mm(A.t(), B)).pow(2) / (torch.trace(torch.mm(A.t(), A)) * torch.trace(torch.mm(B.t(), B)))).clamp(1e-8))

'''
Tests
'''
if __name__ == '__main__':
	set_seed(9001)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Power iteration test
	for _ in range(1000):
		d = 100
		A = torch.randn((d, d), device=device)
		e = np.random.uniform(0.1, 1.10)
		L = torch.linspace(e, 0.01, d, device=device)
		P = torch.mm(torch.mm(A, torch.diag(L)), torch.pinverse(A))

		prec = 1e-4
		np_e_max = np.abs(np.linalg.eigvals(P.cpu().numpy())).max()
		pwr_e_max = spectral_radius(P, eps=prec).item()
		print('True:', e, 'numpy:', np_e_max, 'pwr_iter:', pwr_e_max)
		assert np.abs(np_e_max - pwr_e_max) <= prec