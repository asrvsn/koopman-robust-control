''' Boundary helper functions ''' 

import torch

def rect_boundary(vmin=-float('inf'), vmax=float('inf')):
	'''
	Rectangular boundary imposed on parameter elements.
	'''
	assert vmin < vmax
	def boundary(params: tuple, momentum: tuple, step: float):
		p, m = params[0], momentum[0] # TODO generalize to n-d
		p_cand = p + step*m
		p_max, p_min = p_cand.max(), p_cand.min()
		m_para = torch.zeros_like(m, device=m.device)

		if p_max > vmax:
			grad = torch.autograd.grad(p_max, p_cand)[0]
			m_para = m_para + ((m.t()@grad) / (grad.t()@grad)) * grad
		if p_min < vmin:
			grad = torch.autograd.grad(p_min, p_cand)[0]
			m_para = m_para + ((m.t()@grad) / (grad.t()@grad)) * grad

		if (m_para != 0).any():
			p_cand = p.clamp(min=vmin, max=vmax)
			m_refl = m - 2*m_para
			return ((p_cand.detach().requires_grad_(),), (m_refl.detach().requires_grad_(),)) # Avoid building graph across successive reflections
		return None
	return boundary

def fn_boundary():
	'''
	1D boundary imposed on differentiable scalar function of parameters.
	'''
	pass