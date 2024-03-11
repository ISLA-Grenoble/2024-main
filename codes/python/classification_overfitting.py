import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
    X_00 = mvn.rvs(mean=m_01, cov=C_01, size=int(n_samples/2))
    X_01 = mvn.rvs(mean=m_02, cov=C_02, size=int(n_samples/2))
    X_1 = mvn.rvs(mean=m_1, cov=C_1, size=n_samples)
    X = np.concatenate([X_00, X_01, X_1])
    y = np.array(n_samples*[0] + n_samples*[1])

    return X, y


def get_bayes_boundary(n_grid=51):

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

    pdf_00 = mvn.pdf(x=Z, mean=m_01, cov=C_01)
    pdf_01 = mvn.pdf(x=Z, mean=m_02, cov=C_02)
    pdf_1 = mvn.pdf(x=Z, mean=m_1, cov=C_1)
    t = (0.5 * pdf_00 + 0.5 * pdf_01) - pdf_1
    t = np.reshape(t, [n_grid, n_grid])

    return Z1, Z2, t


def bayes_classifier(x):

    # get the simulation model parameters
    parameters = get_dataset_parameters()
    m_01 = parameters['m_01']
    C_01 = parameters['C_01']
    m_02 = parameters['m_02']
    C_02 = parameters['C_02']
    m_1 = parameters['m_1']
    C_1 = parameters['C_1']

    pdf_00 = mvn.pdf(x=x, mean=m_01, cov=C_01)
    pdf_01 = mvn.pdf(x=x, mean=m_02, cov=C_02)
    pdf_1 = mvn.pdf(x=x, mean=m_1, cov=C_1)

    y_pred = (pdf_1 > (0.5 * pdf_00 + 0.5 * pdf_01)).astype(int)

    return y_pred


def select_mixture(x):
    ch = np.zeros(10)
    ch[np.random.choice(10, 1)] = 1
    return np.dot(x, ch)


def generate_dataset_esl(n_samples):

    filename = "esl_example_means.csv"
    df = pd.read_csv(filename, header=0, index_col=0, names=['X1', 'X2'])
    means_0 = df.values[:10, :]
    means_1 = df.values[10:, :]

    x_0 = np.zeros((n_samples, 2))
    x_1 = np.zeros((n_samples, 2))
    for i in range(n_samples):
        k_0 = np.random.choice(10, 1)[0]
        x_0[i, :] = mvn.rvs(mean=means_0[k_0], size=1)
        k_1 = np.random.choice(10, 1)[0]
        x_1[i, :] = mvn.rvs(mean=means_1[k_1], size=1)

    X = np.concatenate([x_0, x_1])
    y = np.array(n_samples*[0] + n_samples*[1])
    idx = np.random.choice(2*n_samples, 2*n_samples, replace=False)

    return X[idx], y[idx]


def make_figure_knn_overfitting(eps):

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

    X_test, y_test = generate_dataset(n_samples=5_000, eps=1.0)
    y_pred = bayes_classifier(X_test)
    bayes_error = 1-np.mean(y_pred == y_test)

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
    ax.axhline(y=bayes_error, ls='--', c='k', lw=2.0, label='bayes')
    ax.set_xlim(k_list[-1], k_list[0])
    ax.legend()
    ax.set_xlabel('K')
    ax.set_ylabel('Error rate')
    ax.set_xscale('log')
    ax.set_title('Error curves for k-NN')

    return fig, ax


def make_figure_decisiontree_overfitting(eps):

    eps = 1.0
    scores_train_mc = []
    scores_test_mc = []
    for rzt in range(10):
        X_train, y_train = generate_dataset(n_samples=100, eps=eps, seed=rzt)
        X_test, y_test = generate_dataset(n_samples=5_000, eps=eps, seed=rzt)
        scores_train = []
        scores_test = []
        d_list = np.arange(1, 7+1)
        for d in tqdm(d_list):
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=d)
            clf.fit(X_train, y_train)
            scores_train.append(1 - clf.score(X_train, y_train))
            scores_test.append(1 - clf.score(X_test, y_test))
        scores_train_mc.append(np.array(scores_train))
        scores_test_mc.append(np.array(scores_test))

    X_test, y_test = generate_dataset(n_samples=5_000, eps=1.0)
    y_pred = bayes_classifier(X_test)
    bayes_error = 1-np.mean(y_pred == y_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(d_list,
            np.mean(scores_train_mc, axis=0),
            lw=2.0,
            c='C0',
            label='train')
    ax.scatter(d_list,
               np.mean(scores_train_mc, axis=0),
               s=80,
               c='C0')
    ax.plot(d_list,
            np.mean(scores_test_mc, axis=0),
            lw=2.0,
            c='C1',
            label='test')
    ax.scatter(d_list,
               np.mean(scores_test_mc, axis=0),
               s=80,
               c='C1')
    ax.axhline(y=bayes_error, ls='--', c='k', lw=2.0, label='bayes')
    ax.legend()
    ax.set_xlabel('Tree depth')
    ax.set_ylabel('Error rate')
    ax.set_title('Error curves for Decision Trees')

    return fig, ax


eps = 1.0

fig, ax = make_figure_knn_overfitting(eps)
filename = f'knn-train-test-curve-eps-{eps}.pdf'
fig.savefig(filename, format='pdf')

fig, ax = make_figure_decisiontree_overfitting(eps)
filename = f'decisiontree-train-test-curve-eps-{eps}.pdf'
fig.savefig(filename, format='pdf')
