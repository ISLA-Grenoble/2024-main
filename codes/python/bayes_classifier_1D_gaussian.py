import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
plt.style.use('plcr_stylesheet.txt')

x = np.linspace(-5, +10, 1000)

p0 = norm.pdf(x, loc=0, scale=1)
p1 = norm.pdf(x, loc=3, scale=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, p0, lw=3.0, c='C0', label=r'$p_0$')
ax.plot(x, p1, lw=3.0, c='C1', label=r'$p_1$')
ax.axvline(x=3/2, ls='--', c='k')
ax.legend()
fig.savefig('bayes-classifier-1D-gaussian.pdf', format='pdf')
