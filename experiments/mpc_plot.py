import torch
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

from sampler.utils import *

name = 'duffing_mpc'
results = hkl.load(f'saved/{name}.hkl')

t = results['t']
u = results['u']
x = results['x']
r = results['r']

fig, axs = plt.subplots(1, 4)
axs[0].plot(t, x[0], color='blue', label='x1')
axs[0].plot(t, r, color='orange', label='reference')

axs[1].plot(t, x[1])

axs[2].plot(x[0], x[1])

axs[3].plot(t, u)

plt.legend()

plt.show()
