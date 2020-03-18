import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from sampler.utils import *
from sampler.hmc import hamiltonian, gibbs, leapfrog, accept

# def find_initial_step(init_params):

def adapt_step(
      rho: float, t: int, step_size_init: float, H_t: float, eps_bar: float, target_accept_rate: float, \
      gamma=0.05, t0=10, kappa=0.75
    ):
  ''' 
  From Cobb et al, https://github.com/AdamCobb/hamiltorch/blob/master/hamiltorch/samplers.py#L292 
  Default values: p. 16 https://arxiv.org/pdf/1111.4246.pdf
  '''
  t = t + 1
  alpha = min(1., np.exp(rho))
  mu = np.log(step_size_init)
  H_t = (1 - (1/(t + t0)))*H_t + (1/(t + t0))*(target_accept_rate - alpha)
  x_t1 = mu - (t**0.5)/gamma * H_t
  step_size = np.exp(x_t1)
  x_t1_bar = (t**-kappa)*x_t1 + (1 - t**-kappa)*np.log(eps_bar)
  eps_bar = np.exp(x_t1_bar)

  return step_size, eps_bar, H_t 

def sample( 
      n_samples: int, init_params: tuple, potential: Callable, boundary: Callable,
      step_size_init=0.03, n_leapfrog=10, n_burn=30, target_accept_rate=0.75
    ):
  '''
  NUTS HMC 

  potential: once-differentiable potential function 
  boundary: boundary condition which returns either None or (boundary position, reflected momentum)
  step_size_init: initial step size for NUTS sampler
  n_leapfrog: steps per proposal
  n_burn: burn-in-time for step size adaptation
  '''
  assert n_burn > 0

  step_size = step_size_init
  H_t = 0
  eps_bar = 1.0

  params = tuple(x.clone().requires_grad_() for x in init_params)
  ret_params = []
  t = 0

  while len(ret_params) < n_samples:
    print('Current step size: ', step_size)
    momentum = gibbs(params)
    h_old = hamiltonian(params, momentum, potential)
    params, momentum = leapfrog(params, momentum, potential, boundary, n_leapfrog, step_size)

    params = tuple(w.detach().requires_grad_() for w in params)
    h_new = hamiltonian(params, momentum, potential)

    if accept(h_old, h_new) and t > n_burn:
      ret_params.append(params)
      print('Accepted')

    if t < n_burn:
      rho = min(0., float(h_old - h_new))
      step_size, eps_bar, H_t = adapt_step(rho, t, step_size_init, H_t, eps_bar, target_accept_rate)

    elif t == n_burn: 
      step_size = eps_bar
      print('Final adapted step size:', step_size)

    t += 1

  ratio = n_samples / (t - n_burn)
  ret_params = list(map(lambda p: tuple(map(lambda x: x.detach(), p)), ret_params))
  return ret_params, ratio, step_size


if __name__ == '__main__':
  # TODO tests
  pass