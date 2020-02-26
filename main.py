import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from pf import *
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('PF vs. Frobenius norm test - van der Pol')

# Init features
obs = PolynomialObservable(3, 2)
# TODO: ensure observables properly normalized to prevent kernel blowup

# Initialize kernel
d, m, T = obs.k, 2, 3
K = PFKernel(device, d, m)

mu_0 = 0
X, Y = vdp.dataset(mu_0)
P_0, _ = edmd(X, Y, obs)
P_0 = P_0.to(device)

n = 5
results = np.empty((n, 2))
mu_rng = np.linspace(0, 2, n) + mu_0

for i, mu in enumerate(mu_rng):
	X, Y = vdp.dataset(mu)
	P, _ = edmd(X, Y, obs)
	P = P.to(device)
	d_pf = K(P_0, P, T, normalize=True)
	d_fro = torch.norm(P - P_0)
	results[i] = [d_pf.item(), d_fro.item()]

print(results)

plt.figure()
plt.title('P-F distance vs. mu')
plt.plot(mu_rng, results[:, 0])
plt.figure()
plt.title('Frobenius distance vs. mu')
plt.plot(mu_rng, results[:, 1])

plt.show()
