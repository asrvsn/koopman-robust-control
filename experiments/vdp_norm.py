import torch
import matplotlib.pyplot as plt

from sampler.features import *
from sampler.kernel import *
from sampler.operators import *
from sampler.hmc import pdf
from sampler.utils import set_seed
import systems.vdp as vdp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('PF vs. Frobenius norm test - van der Pol')

set_seed(9001)

# Init features
p, d, k = 10, 2, 20
obs = PolynomialObservable(p, d, k)

# Initialize kernel
m, T = 2, 10
K = PFKernel(device, k, m, T)

mu_0 = 0
X, Y = vdp.dataset(mu_0)
PsiX, PsiY = obs(X), obs(Y)
P_0 = dmd(PsiX, PsiY)
P_0 = P_0.to(device)

n = 70
results = np.full((n, 2), np.nan)
mu_rng = np.linspace(0, 3, n) + mu_0

for i, mu in enumerate(mu_rng):
	X, Y = vdp.dataset(mu)
	P = edmd(X, Y, obs)
	P = P.to(device)
	d_pf = K(P_0, P, normalize=True)
	d_fro = torch.norm(P - P_0)
	results[i] = [d_pf.item(), d_fro.item()]

print(results)

p = torch.log(pdf(torch.from_numpy(results[:, 0]), 1))
print(p)

plt.figure()
plt.title('Kernel distance vs. mu')
plt.yscale('log')
plt.plot(mu_rng, results[:, 0])
plt.figure()
plt.title('Probability vs. mu')
plt.plot(mu_rng, p)
plt.figure()
plt.title('Frobenius distance vs. mu')
plt.plot(mu_rng, results[:, 1])

plt.show()
