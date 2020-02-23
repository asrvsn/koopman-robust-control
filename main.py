import torch

from kernel import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize kernel
d, m, T = 50, 2, 5
ker = PFKernel(device, d, m)

if __name__ == '__main__':
	print('Zero test')
	A = torch.randn(d, d, device=device)
	x = ker(A, A, T, normalize=True)
	print('K(A, A) = ', x.item())

	print('Nonzero test')
	A, B = torch.randn(d, d, device=device), torch.randn(d, d, device=device)
	x = ker(A, B, T, normalize=True)
	print('K(A, B) = ', x.item())