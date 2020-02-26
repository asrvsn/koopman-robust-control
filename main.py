import torch
import matplotlib.pyplot as plt

from features import *
from kernel import *
from pf import *
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('PF vs. Frobenius norm test - van der Pol')

# Init features
p, d, k = 3, 2, 5
obs = PolynomialObservable(p, d, k)

# Initialize kernel
m, T = 2, 3
K = PFKernel(device, k, m)

mu_0 = 0
X, Y = vdp.dataset(mu_0)
P_0, _ = edmd(X, Y, obs)
P_0 = P_0.to(device)

n = 30
results = np.full((n, 2), np.nan)
mu_rng = np.linspace(0, 3, n) + mu_0

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
