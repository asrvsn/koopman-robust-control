import numpy as np
import matplotlib.pyplot as plt
import pdb

A = np.array([[1.1,0.75],[-3.25, -0.3]]) / 2 # Nominal: spiral source
U = []
delta = 1.0
for _ in range(1000):
	D = np.random.normal(scale=delta, size=(2,2))
	if np.linalg.norm(D) <= delta:
		U.append(A+D)

U = np.array(U)
tr = np.trace(U, axis1=1, axis2=2)
det = np.linalg.det(U)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axis('off')
xaxis = np.linspace(-2, 2, 100)
yaxis = xaxis**2 / 4
plt.plot(xaxis, np.zeros(xaxis.shape), color='black')
plt.plot(np.zeros(xaxis.shape), xaxis, color='black')
plt.plot(xaxis, yaxis, color='black')
plt.scatter(tr, det, color='orange')
plt.scatter(np.trace(A), np.linalg.det(A), color='blue')
plt.tight_layout()
# plt.show()
fig = plt.gcf()
fig.savefig('tdplot.png', bbox_inches='tight', pad_inches=0)