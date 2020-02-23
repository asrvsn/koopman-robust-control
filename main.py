import torch

from kernel import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize kernel
d, m, T = 10, 2, 100
K = PFKernel(device, d, m)

if __name__ == '__main__':
	print('Zero test')
	A = torch.randn(d, d, device=device)
	x = K(A, A, T, normalize=True)
	print('K(A, A) = ', x.item())

	print('Nonzero test')
	A, B = torch.randn(d, d, device=device), torch.randn(d, d, device=device)
	x = K(A, B, T, normalize=True)
	print('K(A, B) = ', x.item())

	print('Operator validation')
	A, B = torch.eye(d), torch.randn(d, d, device=device)
	P = torch.randn(d, d, device=device).clamp(0)
	P = P.t() / P.sum(1)
	print('Identity operator is valid: ', PFKernel.validate(A))
	print('Random operator is valid:', PFKernel.validate(B))
	print('Stochastic operator is valid:', PFKernel.validate(P))