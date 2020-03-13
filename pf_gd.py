'''
Perron-Frobenius distance gradient descent test
'''

import matplotlib.pyplot as plt
import torch

from kernel import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)

d = 20
A = torch.randn((d, d), device=device)
L = torch.linspace(1.0, 0.05, d, device=device)
P0 = torch.mm(torch.mm(A, torch.diag(L)), torch.pinverse(A))

print('Spectral radius:', spectral_radius(P0))
assert spectral_radius(P0) <= 1.0

A = torch.randn((d, d), device=device)
L = torch.linspace(1.0, 0.05, d, device=device)
P1 = torch.mm(torch.mm(A, torch.diag(L)), torch.pinverse(A))

print('Spectral radius:', spectral_radius(P1))
assert spectral_radius(P1) <= 1.0

P0, P1 = P0.detach().requires_grad_(), P1.detach().requires_grad_()

m, T = 2, 6 # T too large will destabilize the gradient descent
K = PFKernel(device, d, m, T, use_sqrt=False)

eps = 1e-3
step = 1e-3
d = K(P0, P1, normalize=True)
print(d.item())

i, n = 0, 1000
loss = []

while d > eps and i < n:
	(dP,) = torch.autograd.grad(d, P1)
	P1 = P1 - step*dP
	d = K(P0, P1, normalize=True)
	print(d.item())
	loss.append(d.item())
	i += 1

plt.plot(loss)
plt.show()