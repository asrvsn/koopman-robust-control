from typing import Callable
import numpy as np
import random
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
inf = torch.Tensor([float('inf')]).to(device)

def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def zip_with(X: tuple, Y: tuple, f: Callable):
	return tuple(f(x,y) for (x,y) in zip(X, Y))

def eig(A, v):
	return torch.mm(v.t(), torch.mm(A, v))

def spectral_radius(A: torch.Tensor, eps=1e-3):
	d = A.shape[0]
	v = torch.ones((d,1), device=A.device) / np.sqrt(d)
	ev = eig(A, v)
	ev_new = inf

	while torch.abs(ev - ev_new) > eps:
		ev = ev_new
		Av = torch.mm(A, v)
		v_new = Av / torch.norm(Av)
		ev_new = eig(A, v_new)

	return ev_new