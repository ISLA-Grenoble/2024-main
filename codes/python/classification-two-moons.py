from sklearn.datasets import make_moons, make_classification
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from functools import partial


plt.style.use('plcr_stylesheet.txt')


def make_figures_decision_boundaries(clf_dict):

    for clf_method in clf_dict.keys():

        clf = clf_dict[clf_method]

        fig, ax = plt.subplots(figsize=(13.0, 4.0), ncols=3, sharey=True)
        plt.subplots_adjust(wspace=0.15)
        random_state_list = [42, 52, 90]

        for i, axi in enumerate(ax):

            seed_i = random_state_list[i]

            X_train, y_train = make_moons(
                n_samples=200, noise=0.25, random_state=seed_i)
            X_test, y_test = make_moons(
                n_samples=1000, noise=0.25, random_state=666)
            clf.fit(X_train, y_train)

            for c in np.unique(y_test):

                choice = (y_test == c)
                axi.scatter(X_test[choice, 0], X_test[choice, 1], s=20)
                DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X_test,
                    plot_method='contour',
                    ax=axi,
                    levels=[0.5],
                    linewidths=2.0,
                    colors='k',
                    linestyles='dashed'
                )
                axi.set_xlim(-2.0, +3.0)
                axi.set_ylim(-2.5, +2.5)
                axi.set_title(f'seed = {seed_i}')

        fig.savefig(f'classification-two-moons_{clf_method}.pdf', format='pdf')


clf_dict = {}
clf_dict['decision_tree'] = DecisionTreeClassifier(
    criterion='entropy', max_depth=6)
clf_dict['log_reg'] = LogisticRegression()
clf_dict['knn'] = KNeighborsClassifier(n_neighbors=10)
clf_dict['random_forest'] = RandomForestClassifier(
    n_estimators=100, criterion='entropy')

make_figures_decision_boundaries(clf_dict=clf_dict)
