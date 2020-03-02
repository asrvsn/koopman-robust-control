import torch

class PFKernel:
	def __init__(self, device: torch.device, d: int, m: int, T: int):
		'''
		TODOs: 
		* allocate less memory
		* possibly use torch.cartesian_product instead of repeat & meshgrid
		'''
		self.device = device
		self.d = d
		self.T = T
		with torch.no_grad():
			index = torch.arange(d, device=device).expand(m, -1)
			product = torch.meshgrid(index.unbind())
			mask = torch.triu(torch.ones(d, d, device=device), diagonal=1).bool() # upper triangle indices
			subindex = (torch.masked_select(x, mask) for x in product) # select unique combinations
			subindex = torch.stack(tuple(subindex)).t() # submatrix index I in paper
			n = subindex.shape[0]
			self.subindex_row = subindex.unsqueeze(1).repeat(1, n, 1).view(n*n, -1) # row selector
			self.subindex_col = subindex.unsqueeze(1).repeat(n, 1, 1).view(n*n, -1) 
			self.subindex_col = self.subindex_col.t().repeat(m, 1, 1).permute(2, 0, 1) # column selector different due to Torch API

	def __call__(self, P1: torch.Tensor, P2: torch.Tensor, normalize=False):
		# P1, P2 must be square and same shape
		# assert P1.shape == (self.d, self.d) and P2.shape == P1.shape
		eps = 1e-6
		if normalize:
			# d = torch.sqrt(1 - self.__call__(P1, P2).pow(2) / (self.__call__(P1, P1) * self.__call__(P2, P2)))
			d = 1 - self.__call__(P1, P2).pow(2) / (self.__call__(P1, P1) * self.__call__(P2, P2))
			# d = d.clamp(1e-3)
			print(d.item())
			if torch.isinf(d).item():
				print('inf!')
			if torch.isnan(d).item():
				print('nan!')
			return d
		else:
			sum_powers = torch.eye(self.d, device=self.device)
			# print(torch.isnan(P1).any().item(), torch.isnan(P2).any().item())
			# TODO: fix nan here
			for _ in range(1, self.T):
				sum_powers = sum_powers + torch.mm(torch.mm(P1, sum_powers), P2.t())
			# print(torch.isnan(sum_powers).any().item())
			# print(torch.isinf(sum_powers).any().item())
			submatrices = torch.gather(sum_powers[self.subindex_row], 2, self.subindex_col) 
			# print(submatrices.max().item())
			result = submatrices.det().sum()
			# result = result.clamp(1e-6, float('inf'))
			# print(result.item())
			return result

	@staticmethod
	def validate(P: torch.Tensor):
		# Kernel only valid for operators with eigenvalues on unit circle.
		with torch.no_grad():
			(eig, _) = torch.eig(P)
			re, im = eig[:, 0], eig[:, 1]
			norm = torch.sqrt(re.pow(2) + im.pow(2))
			return (norm <= 1.0).all().item()

'''
Tests for P-F kernel
'''
if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	torch.autograd.set_detect_anomaly(True)

	# Initialize kernel
	d, m, T = 20, 2, 3
	K = PFKernel(device, d, m, T)

	print('Zero test')
	A = torch.randn(d, d, device=device)
	x = K(A, A, normalize=True)
	print('K(A, A) = ', x.item())

	print('Nonzero test')
	A, B = torch.randn(d, d, device=device), torch.randn(d, d, device=device)
	x = K(A, B, normalize=True)
	print('K(A, B) = ', x.item())

	print('Operator validation')
	A, B = torch.eye(d), torch.randn(d, d, device=device)
	P = B.clamp(0)
	P = (P.t() / P.sum(1)).t()
	print('Identity operator is valid: ', PFKernel.validate(A))
	print('Random operator is valid:', PFKernel.validate(B))
	print('Stochastic operator is valid:', PFKernel.validate(P))