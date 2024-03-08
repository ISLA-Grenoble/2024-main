import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import pandas as pd


plt.style.use('plcr_stylesheet.txt')


def get_dataset_parameters(eps=1.0):

    # class 0
    theta_01 = np.pi/2.5
    Q_01 = np.array([[np.cos(theta_01), -np.sin(theta_01)],
                    [np.sin(theta_01), np.cos(theta_01)]])
    D_01 = np.diag([0.25, 1.0])
    m_01, C_01 = np.array([0.0, 2.0]), Q_01 @ D_01 @ Q_01.T
    theta_02 = -np.pi/2
    Q_02 = np.array([[np.cos(theta_02), -np.sin(theta_02)],
                    [np.sin(theta_02), np.cos(theta_02)]])
    D_02 = np.diag([0.35, 1.2])
    m_02, C_02 = np.array([0.5, -1.0]), Q_02 @ D_02 @ Q_02.T
    # class 1
    theta_1 = -np.pi/3
    Q_1 = np.array([[np.cos(theta_1), -np.sin(theta_1)],
                    [np.sin(theta_1), np.cos(theta_1)]])
    D_1 = np.diag([0.5, 1.0])
    m_1, C_1 = np.array([eps*2.25, 0.0]), Q_1 @ D_1 @ Q_1.T

    parameters = {}
    parameters['m_01'] = m_01
    parameters['C_01'] = C_01
    parameters['m_02'] = m_02
    parameters['C_02'] = C_02
    parameters['m_1'] = m_1
    parameters['C_1'] = C_1

    return parameters


def generate_dataset(n_samples=100, eps=1.0, seed=42):

    np.random.seed(seed)

    # get the simulation model parameters
    parameters = get_dataset_parameters(eps=eps)
    m_01 = parameters['m_01']
    C_01 = parameters['C_01']
    m_02 = parameters['m_02']
    C_02 = parameters['C_02']
    m_1 = parameters['m_1']
    C_1 = parameters['C_1']

    # sample points from each class and join them into a single array
    X_00 = multivariate_normal.rvs(mean=m_01, cov=C_01, size=int(n_samples/2))
    X_01 = multivariate_normal.rvs(mean=m_02, cov=C_02, size=int(n_samples/2))
    X_1 = multivariate_normal.rvs(mean=m_1, cov=C_1, size=n_samples)
    X = np.concatenate([X_00, X_01, X_1])
    y = np.array(n_samples*[0] + n_samples*[1])

    return X, y


def get_bayes_boundary(n_grid=51, seed=42):

    # get the simulation model parameters
    parameters = get_dataset_parameters()
    m_01 = parameters['m_01']
    C_01 = parameters['C_01']
    m_02 = parameters['m_02']
    C_02 = parameters['C_02']
    m_1 = parameters['m_1']
    C_1 = parameters['C_1']

    z1 = np.linspace(-5, +10, n_grid)
    z2 = np.linspace(-2.5, +5, n_grid)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.stack([Z1.flatten(), Z2.flatten()]).T

    pdf_00 = multivariate_normal.pdf(x=Z, mean=m_01, cov=C_01)
    pdf_01 = multivariate_normal.pdf(x=Z, mean=m_02, cov=C_02)
    pdf_1 = multivariate_normal.pdf(x=Z, mean=m_1, cov=C_1)
    t = (0.5 * pdf_00 + 0.5 * pdf_01) - pdf_1
    t = np.reshape(t, [n_grid, n_grid])

    return Z1, Z2, t


def select_mixture(x):
    ch = np.zeros(10)
    ch[np.random.choice(10, 1)] = 1
    return np.dot(x, ch)


def generate_dataset_esl(n_samples):

    filename = "esl_example_means.csv"
    df = pd.read_csv(filename, header=0, index_col=0, names=['X1', 'X2'])
    means_0 = df.values[:10, :]
    means_1 = df.values[10:, :]

    # x_0 = np.zeros((n_samples, 2, 10))
    # x_1 = np.zeros((n_samples, 2, 10))
    # for i in range(10):
    #     x_0[:, :, i] = multivariate_normal.rvs(mean=means_0[i], size=n_samples)
    #     x_1[:, :, i] = multivariate_normal.rvs(mean=means_1[i], size=n_samples)
    # x_0 = np.apply_along_axis(select_mixture, 2, x_0)
    # x_1 = np.apply_along_axis(select_mixture, 2, x_1)

    x_0 = np.zeros((n_samples, 2))
    x_1 = np.zeros((n_samples, 2))
    for i in range(n_samples):
        k = np.random.choice(10, 1)[0]
        x_0[i, :] = multivariate_normal.rvs(mean=means_0[k], size=1)
        x_1[i, :] = multivariate_normal.rvs(mean=means_1[k], size=1)

    X = np.concatenate([x_0, x_1])
    y = np.array(n_samples*[0] + n_samples*[1])
    idx = np.random.choice(2*n_samples, 2*n_samples, replace=False)

    return X[idx], y[idx]


eps = 0.25
scores_train_mc = []
scores_test_mc = []
for rzt in range(10):
    X_train, y_train = generate_dataset(n_samples=100, eps=eps, seed=rzt)
    X_test, y_test = generate_dataset(n_samples=5_000, eps=eps, seed=rzt)
    scores_train = []
    scores_test = []
    k_list = np.arange(1, 150+1)
    for k in tqdm(k_list):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        scores_train.append(1 - clf.score(X_train, y_train))
        scores_test.append(1 - clf.score(X_test, y_test))
    scores_train_mc.append(np.array(scores_train))
    scores_test_mc.append(np.array(scores_test))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(k_list,
        np.mean(scores_train_mc, axis=0),
        lw=2.0,
        c='C0',
        label='train')
ax.plot(k_list,
        np.mean(scores_test_mc, axis=0),
        lw=2.0,
        c='C1',
        label='test')
ax.set_xlim(k_list[-1], k_list[0])
ax.legend()
ax.set_xscale('log')
filename = f'knn-train-test-curve-eps-{eps}.pdf'
fig.savefig(filename, format='pdf')

# X, y = generate_dataset_esl(n_samples=500)
# fig, ax = plt.subplots(figsize=(7.0, 6.5))
# colors = {0: 'C0', 1: 'C1'}
# for cl in np.unique(y):
#     ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)
# # ax.set_xlim(-2.5, 5.5)
# # ax.set_ylim(-2.5, 3.0)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(1.2)
# ax.set_xlabel(r'$X_1$', fontsize=16)
# ax.set_ylabel(r'$X_2$', fontsize=16)
# fig.show()
