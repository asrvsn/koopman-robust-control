import torch
import numpy as np
import scipy.linalg as linalg

from sampler.utils import *
from sampler.dynamics import *
from systems.lti2x2 import systems

set_seed(9001)

A = systems['spiral sink']
model = torch.from_numpy(linalg.expm(A)).float()
print(model)

# # Initial conditions
# N = 10
# ics = make_ics(N, model, (1e-2, 1e-2), 3e-5, 50, debug=True)

# Perturbations
n_samples = 10
n_split = 5
beta = 5
samples = perturb(n_samples, model, beta, r_div=(1e-2,1e-2), r_step=3e-5, n_split=n_split, debug=True)

print('Got samples:', len(samples))