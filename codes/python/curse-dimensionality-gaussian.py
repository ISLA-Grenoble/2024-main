import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plcr_stylesheet.txt')

r = np.linspace(0, 6, 100)
fig, ax = plt.subplots(figsize=(8.0, 4.0))
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.10, right=0.95)
for k in [1, 2, 10]:
    pdf = np.power(r, k-1) * np.exp(-r**2/2)
    dr = r[1] - r[0]
    pdf_k = pdf / (np.sum(pdf) * dr)
    ax.plot(r, pdf_k, label=f'{k}', lw=3.0)
ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$r$', fontsize=14)
ax.set_ylabel(r'$p(r)$', fontsize=14)
ax.legend()
fig.savefig('curse-dimensionality-gaussian.pdf', format='pdf')
