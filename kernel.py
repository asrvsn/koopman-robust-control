import torch

class PFKernel:
	def __init__(self, device: torch.device, d: int, m: int):
		self.device = device
		self.d = d
		with torch.no_grad():
			index = torch.arange(d, device=device).expand(m, -1)
			product = torch.meshgrid(index.unbind())
			mask = torch.triu(torch.ones(d, d, device=device), diagonal=1).bool() # upper triangle indices
			subindex = (torch.masked_select(x, mask) for x in product) # select unique combinations
			subindex = torch.stack(tuple(subindex)).t() # submatrix index I in paper
			n = subindex.shape[0]
			self.subindex_row = subindex.unsqueeze(1).repeat(1, n, 1).view(n*n, -1) # row selector
			subindex_col = subindex.unsqueeze(1).repeat(n, 1, 1).view(n*n, -1) 
			self.subindex_col = subindex_col.t().repeat(m, 1, 1).permute(2, 0, 1) # column selector different due to Torch API

	def __call__(self, P1: torch.Tensor, P2: torch.Tensor, T: int, normalize=False):
		# P1, P2 must be square and same shape
		# assert P1.shape == (self.d, self.d) and P2.shape == P1.shape
		if normalize:
			return torch.sqrt(1 - self.__call__(P1, P2, T).pow(2) / (self.__call__(P1, P1, T) * self.__call__(P2, P2, T)))
		else:
			sum_powers = torch.eye(self.d, device=self.device)
			for _ in range(1, T):
				sum_powers += torch.mm(torch.mm(P1, sum_powers), P2.t())
			submatrices = torch.gather(sum_powers[self.subindex_row], 2, self.subindex_col) 
			result = submatrices.det().sum()
			return result
