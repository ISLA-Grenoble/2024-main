import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# import stylesheet
plt.style.use('plcr_stylesheet.txt')

# fix the seed
np.random.seed(42)

# fix the number of samples to consider for each xi
Ni = 100

# fix the number of x values to consider
Ns = 21

# choose the values for the xi
x = np.linspace(1, 5, Ns)

# populate the dataframe with realizations of y
df = []
for i in range(Ni):
    ei = (3*x + 1) * 0.25 * np.random.randn(Ns)
    yi = (x-1)**2 + ei
    dfi = pd.DataFrame()
    dfi['x'] = x
    dfi['y'] = yi
    df.append(dfi)
df = pd.concat(df)

# choose the conditioning observation
xcond = 3

# get the samples related to the specific conditioning
ycond = df['y'][df['x'] == 3].values

# regress
model = sm.OLS(df['y'], sm.add_constant(df['x']))
res = model.fit()
print(res.summary())

# plot
fig, ax = plt.subplots(figsize=(10.9, 5.3))
ax.scatter(df['x'], df['y'], s=10, c='gray')
# ax.axhline(y=np.mean(df['y']), c='C0')
# ax.scatter(xcond * np.ones(Ni), ycond, s=10, c='C3')
# ax.scatter(xcond, 4, s=100, c='C0')
# ax.plot(x, (x-1)**2, lw=2.0, c='C0')
ax.plot(x, res.params['const'] + res.params['x'] * x, lw=2.0, c='C0')
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_xlim(1, 5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(-5, +25)
plt.savefig('cond_expect_fig.pdf', format='pdf')
fig.show()
