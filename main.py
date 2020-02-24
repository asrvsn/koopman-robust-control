import torch

from features import *
from kernel import *
from pf import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize kernel
d, m, T = 10, 2, 10
K = PFKernel(device, d, m)

