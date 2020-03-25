import matplotlib.pyplot as plt
import scipy.stats as stats
import hickle as hkl
import random

results = hkl.load('saved/duffing.hkl')
nominal = torch.from_numpy(results['nominal']).float()
posterior = results['posterior']
samples = [torch.from_numpy(s).float() for s in results['samples']]
step = results['step']
beta = results['beta']

# Visualize perturbations

mu = 3.0
n_vdp = 6000
b_vdp = 40
skip_vdp = int(n_vdp * 8 / b_vdp) # start on limit cycle
X, Y = vdp.dataset(mu, n=n_vdp, b=b_vdp, skip=skip_vdp)
t = int(n_vdp/2)

fig, axs = plt.subplots(1, 2)

axs[0].set_xlim(left=-7.0, right=7.0)
axs[0].set_ylim(bottom=-7.0, top=7.0)
axs[0].set_title(f'samples, step={step}')

for Pn in random.choices(samples, k=20):
	Zn = obs.extrapolate(Pn, X, t)
	axs[0].plot(Zn[0], Zn[1], color='grey', alpha=0.4)

Z0 = obs.extrapolate(nominal.cpu(), X, t)
axs[0].plot(Z0[0], Z0[1], color='blue')
axs[0].plot([X[0][0]], [X[1][0]], marker='o', color='blue')

# Sample distribution

axs[1].hist(posterior, bins=int(len(posterior)/4), density=True)
d = np.linspace(0, 1, 100)
p = stats.beta.pdf(d, 1.0, beta)
axs[1].plot(d, p)
axs[1].set_title(f'Posterior distribution, beta={beta}')

# Closeup

# t = 300

# plt.figure()
# for Pn in samples:
# 	Zn = obs.extrapolate(Pn.cpu(), X, t+50)
# 	plt.plot(Zn[0], Zn[1], color='grey', alpha=0.3)

# plt.xlim(left=X[0][t], right=X[0][0] + 0.05)
# plt.ylim(bottom=X[1][t], top=X[1][0] + 0.05)
# plt.title(f'beta={beta}, step={hmc_step}, T={T}, distance={"Frobenius" if baseline else "Kernel"}')
# Z0 = obs.extrapolate(nominal.cpu(), X, t+100)
# plt.plot(Z0[0], Z0[1], label='Nominal')

# plt.plot([X[0][0]], [X[1][0]], marker='o', color='blue')
# plt.legend()

plt.show()