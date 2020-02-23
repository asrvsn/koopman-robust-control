import torch

from kernel import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize kernel
d, m, T = 50, 2, 5
ker = PFKernel(device, d, m)

if __name__ == '__main__':
	for i in range(10):
		A = torch.randn(d, d, device=device)
		x = ker(A, A, T, normalize=True)
		print(x.item())