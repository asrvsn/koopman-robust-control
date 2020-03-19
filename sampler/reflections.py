''' 
Boundary helper functions 
''' 

from typing import Callable
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
			p_cand = p_cand.clamp(min=vmin, max=vmax)
			m_refl = m - 2*m_para
			return ((p_cand.detach().requires_grad_(),), (m_refl.detach().requires_grad_(),)) # Avoid building graph across successive reflections
		return None
	return boundary

def fn_boundary(fn: Callable, vmin=-float('inf'), vmax=float('inf'), boundary_resolution=10):
	'''
	1D boundary imposed on differentiable scalar function of parameters.
	'''
	assert vmin < vmax
	def boundary(params: tuple, momentum: tuple, step: float):
		p, m = params[0], momentum[0] # TODO generalize to n-d
		v = fn(p + m*step)

		if v > vmax:
			micro_step = step/boundary_resolution
			p_cand = p.detach()
			while fn(p) < vmax:
				p_cand += micro_step*m
			p_cand = p_cand.requires_grad_()
			# Reflect along plane orthogonal to gradient
			grad = torch.autograd.grad(fn(p_cand), p_cand)[0]
			m_para = (torch.trace(m.t()@grad) / torch.trace(grad.t()@grad)) * grad
			m_refl = m - 2*m_para
			return ((p_cand.detach().requires_grad_(),), (m_refl.detach().requires_grad_(),))

		elif v < vmin:
			micro_step = step/boundary_resolution
			p_cand = p.detach()
			while fn(p) > vmin:
				p_cand += micro_step*m
			p_cand = p_cand.requires_grad_()
			# Reflect along plane orthogonal to gradient
			grad = -torch.autograd.grad(fn(p_cand), p_cand)[0]
			m_para = (torch.trace(m.t()@grad) / torch.trace(grad.t()@grad)) * grad
			m_refl = m - 2*m_para
			return ((p_cand.detach().requires_grad_(),), (m_refl.detach().requires_grad_(),))

		return None
	return boundary