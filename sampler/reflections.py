''' 
Boundary helper functions 
''' 

from typing import Callable
import torch

def lp_boundary(lp: float, vmax=float('inf'), delta=1e-2):
	'''
	Boundary imposed on lp norm of parameter elements.
	'''
	def boundary(params: tuple, momentum: tuple, step: float):
		p, m = params[0], momentum[0] # TODO generalize to n-d
		p_cand = p + step*m

		if p_cand.norm(p=lp) > vmax:
			p_cand = p.detach()
			eps = step
			while (p_cand + delta*m).norm(p=lp) <= vmax:
				p_cand += delta*m
				eps -= delta
			p_cand = p_cand.requires_grad_()
			grad = torch.autograd.grad(p_cand.norm(p=lp), p_cand)[0]
			m_para = ((m.t()@grad) / (grad.t()@grad)) * grad
			m_refl = m - 2*m_para
			return (p_cand.detach().requires_grad_(),), (m_refl.detach().requires_grad_(),), eps

		return (p_cand,), momentum, 0

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