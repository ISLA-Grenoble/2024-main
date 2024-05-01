
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

filepath = "./zkcc-77/karate_edges_77.txt"
edges = np.loadtxt(filepath)
filepath = "./zkcc-77/karate_groups.txt"
groups = np.loadtxt(filepath)[:,1].astype(int)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

Neigh = {}
for edge in edges:
    vi, vj = int(edge[0]), int(edge[1])
    if vi not in Neigh:
        Neigh[vi] = []
    if vj not in Neigh:
        Neigh[vj] = []
    Neigh[vi].append(vj)
    Neigh[vj].append(vi)

dis = {}
dis_matrix = np.zeros( (len(Neigh), len(Neigh)) )
for i in Neigh.keys():
    for j in Neigh.keys():
        intersec = list(set(Neigh[i]).intersection(Neigh[j]))
        dis[(i, j)] = len(intersec) / np.sqrt( len(Neigh[i]) * len(Neigh[j]) )
        dis_matrix[i-1, j-1] = dis[(i, j)]

model = AgglomerativeClustering(
    n_clusters=None, affinity="precomputed", distance_threshold=0, linkage="average")
model = model.fit(1-dis_matrix)

# plot_dendrogram(model, color_threshold=0.95, no_labels=False)
# fig, ax = plt.gcf(), plt.gca()
# ax.set_ylim(-0.01, 0.97)
# ax.set_xticklabels(groups, fontsize=12)
# # ax.set_yticklabels([])
# fig.set_size_inches([12.14, 6.66])
# plt.savefig('zachary-agglomerative-dendogram.pdf', format='pdf')

import networkx as nx
G = nx.karate_club_graph()
B = nx.modularity_matrix(G)
B = np.asarray(B)
w, v = np.linalg.eig(B)
w = np.real(w)
v = np.real(v)
idx = w.argsort()[::-1]
w = w[idx]
v = v[:,idx]

communities_spectral = np.sign(v[:,0]).astype(int)
comm1 = set(np.where(communities_spectral > 0)[0])
comm2 = set(np.where(communities_spectral < 0)[0])
partition_spectral = (comm1, comm2)
Q_spectral = nx.community.modularity(G, communities=partition_spectral)
print(Q_spectral)

# communities_colors = {1:'C0', -1:'C1'}
# colors = [communities_colors[comm] for comm in communities]
# nx.draw(G, node_color=colors)
# plt.show()

partition_greedy = nx.community.greedy_modularity_communities(G)
Q_greedy = nx.community.modularity(G, communities=partition_greedy)
print(Q_greedy)