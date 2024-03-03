import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay


plt.style.use('plcr_stylesheet.txt')


def get_dataset_parameters():

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
    m_1, C_1 = np.array([2.25, 0.0]), Q_1 @ D_1 @ Q_1.T

    parameters = {}
    parameters['m_01'] = m_01
    parameters['C_01'] = C_01
    parameters['m_02'] = m_02
    parameters['C_02'] = C_02
    parameters['m_1'] = m_1
    parameters['C_1'] = C_1

    return parameters


def generate_dataset(n_points=100, seed=42):

    np.random.seed(seed)

    # get the simulation model parameters
    parameters = get_dataset_parameters()
    m_01 = parameters['m_01']
    C_01 = parameters['C_01']
    m_02 = parameters['m_02']
    C_02 = parameters['C_02']
    m_1 = parameters['m_1']
    C_1 = parameters['C_1']

    # sample points from each class and join them into a single array
    X_00 = multivariate_normal.rvs(mean=m_01, cov=C_01, size=int(n_points/2))
    X_01 = multivariate_normal.rvs(mean=m_02, cov=C_02, size=int(n_points/2))
    X_1 = multivariate_normal.rvs(mean=m_1, cov=C_1, size=n_points)
    X = np.concatenate([X_00, X_01, X_1])
    y = np.array(n_points*[0] + n_points*[1])

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


def make_figure_01(n_points, seed=42):

    X, y = generate_dataset(n_points, seed)
    clf_knn = KNeighborsClassifier(n_neighbors=15)
    clf_knn.fit(X, y)

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'C0', 1: 'C1'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    Z1, Z2, t = get_bayes_boundary(n_grid=51)
    ax.contour(Z1, Z2, t,
               levels=np.array([0.0]),
               colors='k',
               linestyles='dashed')

    Z = np.stack([Z1.flatten(), Z2.flatten()]).T
    t = clf_knn.predict_proba(Z)[:, 0]
    t = t.reshape(Z1.shape)
    ax.contour(Z1, Z2, t,
               levels=np.array([0.5]),
               colors='C2',
               linestyles='dashed',
               linewidths=2.0)

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)
    ax.plot([], [], c='k', lw=1.5, ls='--', label='Bayes')

    return fig, ax


def make_figure_02a(n_points, seed=42):

    X, y = generate_dataset(n_points, seed)

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'k', 1: 'k'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)
    ax.plot([], [], c='k', lw=1.5, ls='--', label='Bayes')

    return fig, ax


def make_figure_02b(n_points, seed=42):

    X, y = generate_dataset(n_points, seed)

    # choose an arbitrary (and useful) data point
    idx = 30
    xi = X[idx,]
    di = np.sum((X - xi)**2, axis=1)
    idx_neigh = di.argsort()[1:(10+1)]

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'k', 1: 'k'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    ax.scatter(xi[0], xi[1], c='C3', s=125)
    X_neigh, y_neigh = X[idx_neigh], y[idx_neigh]
    ax.scatter(X_neigh[y_neigh == 0, 0],
               X_neigh[y_neigh == 0, 1],
               c='C0', s=40)
    ax.scatter(X_neigh[y_neigh == 1, 0],
               X_neigh[y_neigh == 1, 1],
               c='C1', s=40)

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)
    ax.plot([], [], c='k', lw=1.5, ls='--', label='Bayes')

    return fig, ax


def make_figure_03(n_points, seed=42):

    X, y = generate_dataset(n_points, seed)

    # level 0
    var = X.var(axis=0)
    dim_0 = var.argmax()
    t_0 = np.median(X[:, dim_0])

    # level 1 (up)
    sel_1_up = (X[:, dim_0] < t_0)
    var = X[sel_1_up].var(axis=0)
    dim_1_up = var.argmax()
    t_1_up = np.median(X[sel_1_up, dim_1_up])

    # level 1 (down)
    sel_1_down = (X[:, dim_0] >= t_0)
    var = X[sel_1_down].var(axis=0)
    dim_1_down = var.argmax()
    t_1_down = np.median(X[sel_1_down, dim_1_down])

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'k', 1: 'k'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)

    ax.axvline(x=t_0, ls='--', c='k')
    xthr = (t_0-ax.get_xlim()[0])/np.diff(ax.get_xlim())[0]
    ax.axhline(y=t_1_up, xmin=0, xmax=xthr, ls='--', c='k')
    ax.axhline(y=t_1_down, xmin=xthr, xmax=1.0, ls='--', c='k')

    return fig, ax


def make_figure_04(n_points, seed):

    X, y = generate_dataset(n_points, seed)

    # level 0
    var = X.var(axis=0)
    dim_0 = var.argmax()
    t_0 = np.median(X[:, dim_0])

    # level 1 (up)
    sel_1_up = (X[:, dim_0] < t_0)
    var = X[sel_1_up].var(axis=0)
    dim_1_up = var.argmax()
    t_1_up = np.median(X[sel_1_up, dim_1_up])

    # level 1 (down)
    sel_1_down = (X[:, dim_0] >= t_0)
    var = X[sel_1_down].var(axis=0)
    dim_1_down = var.argmax()
    t_1_down = np.median(X[sel_1_down, dim_1_down])

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'C0', 1: 'C1'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)

    ax.axvline(x=t_0, ls='--', c='C4')
    xthr = (t_0-ax.get_xlim()[0])/np.diff(ax.get_xlim())[0]
    ax.axhline(y=t_1_up, xmin=0, xmax=xthr, ls='--', c='C4')
    ax.axhline(y=t_1_down, xmin=xthr, xmax=1.0, ls='--', c='C4')

    return fig, ax


def make_figure_05(n_points, seed):

    X, y = generate_dataset(n_points, seed)

    # level 0
    var = X.var(axis=0)
    dim_0 = var.argmax()
    t_0 = np.median(X[:, dim_0])

    # level 1 (up)
    sel_1_up = (X[:, dim_0] < t_0)
    var = X[sel_1_up].var(axis=0)
    dim_1_up = var.argmax()
    t_1_up = np.median(X[sel_1_up, dim_1_up])

    # level 1 (down)
    sel_1_down = (X[:, dim_0] >= t_0)
    var = X[sel_1_down].var(axis=0)
    dim_1_down = var.argmax()
    t_1_down = np.median(X[sel_1_down, dim_1_down])

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'C0', 1: 'C1'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    Z1, Z2, t = get_bayes_boundary(n_grid=51)
    ax.contour(Z1, Z2, t,
               levels=np.array([0.0]),
               colors='k',
               linestyles='dashed')        

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)

    ax.axvline(x=t_0, ls='--', c='C2')

    return fig, ax


def make_figure_06(n_points, seed):

    X, y = generate_dataset(n_points, seed)

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    colors = {0: 'C0', 1: 'C1'}
    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=colors[cl], s=30)

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        plot_method='contour',
        response_method='predict',
        ax=ax,
        levels=0,
        colors='C2'
    )

    Z1, Z2, t = get_bayes_boundary(n_grid=51)
    ax.contour(Z1, Z2, t,
               levels=np.array([0.0]),
               colors='k',
               linestyles='dashed')

    ax.set_xlim(-2.5, 5.5)
    ax.set_ylim(-2.5, 3.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.2)
    ax.set_xlabel(r'$X_1$', fontsize=16)
    ax.set_ylabel(r'$X_2$', fontsize=16)

    return fig, ax


make_figure_list = [make_figure_01,
                    make_figure_02a,
                    make_figure_02b,
                    make_figure_04,
                    make_figure_05,
                    make_figure_06]

for i, make_figure in enumerate(make_figure_list):
    fig, ax = make_figure(n_points=100, seed=42)
    filename = f'decision_trees_fig_{i+1:02}.pdf'
    fig.savefig(filename, format='pdf')
