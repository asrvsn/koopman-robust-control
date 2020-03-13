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

def eig(A, v):
	return torch.mm(v.t(), torch.mm(A, v))

def spectral_radius(A: torch.Tensor, eps=1e-6):
	d = A.shape[0]
	v = torch.ones((d,1), device=A.device) / np.sqrt(d)
	ev = eig(A, v)
	ev_new = torch.Tensor([float('inf')]).to(A.device)

	while torch.abs(ev - ev_new) > eps:
		ev = ev_new
		Av = torch.mm(A, v)
		v_new = Av / torch.norm(Av)
		ev_new = eig(A, v_new)

	return ev_new

def deduped_legend():
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())

def euclidean_matrix_kernel(A: torch.Tensor, B: torch.Tensor):
	return 1 - torch.trace(torch.mm(A.t(), B)).pow(2) / (torch.trace(torch.mm(A.t(), A)) * torch.trace(torch.mm(B.t(), B)))