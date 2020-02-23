import torch

from kernel import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize kernel
d, m, T = 3, 2, 1
ker = PFKernel(device, d, m)

if __name__ == '__main__':
	A = torch.randn(d, d, device=device)
	x = ker(A, A, T)
	print(x.item())