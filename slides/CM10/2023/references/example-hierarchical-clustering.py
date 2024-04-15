
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from itertools import chain, combinations

def plot_dendrogram_from_sklearn(model, **kwargs):
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

    fig = plt.gcf()
    return fig

def plot_dendrogram_from_networkx(communities):

    # building initial dict of node_id to each possible subset:
    node_id = 0
    init_node2community_dict = {node_id: communities[0][0].union(communities[0][1])}
    for comm in communities:
        for subset in list(comm):
            if subset not in init_node2community_dict.values():
                node_id += 1
                init_node2community_dict[node_id] = subset

    # turning this dictionary to the desired format in @mdml's answer
    node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
    for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
        for node_id_parent, group in init_node2community_dict.items():
            if len(init_node2community_dict[node_id1].intersection(init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(init_node2community_dict[node_id2]):
                node_id_to_children[node_id_parent].append(node_id1)
                node_id_to_children[node_id_parent].append(node_id2)

    # also recording node_labels dict for the correct label for dendrogram leaves
    node_labels = dict()
    for node_id, group in init_node2community_dict.items():
        if len(group) == 1:
            node_labels[node_id] = list(group)[0]
        else:
            node_labels[node_id] = ''

    # also needing a subset to rank dict to later know within all k-length merges which came first
    subset_rank_dict = dict()
    rank = 0
    for e in communities[::-1]:
        for p in list(e):
            if tuple(p) not in subset_rank_dict:
                subset_rank_dict[tuple(sorted(p))] = rank
                rank += 1
    subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank

    # my function to get a merge height so that it is unique (probably not that efficient)
    def get_merge_height(sub):
        sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
        n = len(sub_tuple)
        other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
        min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
        range = (max_rank-min_rank) if max_rank > min_rank else 1
        return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range

    # finally using @mdml's magic, slightly modified:
    G           = nx.DiGraph(node_id_to_children)
    nodes       = G.nodes()
    leaves      = set( n for n in nodes if G.out_degree(n) == 0 )
    inner_nodes = [ n for n in nodes if G.out_degree(n) > 0 ]

    # Compute the size of each subtree
    subtree = dict( (n, [n]) for n in leaves )
    for u in inner_nodes:
        children = set()
        node_list = list(node_id_to_children[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add( v )
            node_list += node_id_to_children[v]
        subtree[u] = sorted(children & leaves)

    inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last

    # Construct the linkage matrix
    leaves = sorted(leaves)
    index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = node_id_to_children[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
            Z.append([i, j, get_merge_height(subtree[n]), len(z)]) # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1

    # dendrogram
    fig = plt.figure()
    dendrogram(Z, labels=[node_labels[node_id] for node_id in leaves])
    return fig    

Neigh = {}
Neigh[1] = [2, 3, 4]
Neigh[2] = [1, 3]
Neigh[3] = [1, 2, 4]
Neigh[4] = [1, 3, 5, 6]
Neigh[5] = [4, 6, 7, 8]
Neigh[6] = [4, 5, 7, 8]
Neigh[7] = [5, 6, 8, 9]
Neigh[8] = [6, 7, 9]
Neigh[9] = [7]

dis = {}
dis_matrix = np.zeros( (len(Neigh), len(Neigh)) )
for i in Neigh.keys():
    for j in Neigh.keys():
        intersec = list(set(Neigh[i]).intersection(Neigh[j]))
        dis[(i, j)] = len(intersec) / np.sqrt( len(Neigh[i]) * len(Neigh[j]) )
        dis_matrix[i-1, j-1] = dis[(i, j)]

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.matshow(np.zeros( (len(Neigh), len(Neigh)) ), cmap='Greys')
# ax.set_xticks(range(1, len(Neigh)+1))
# ax.set_yticks(range(1, len(Neigh)+1))
# ax.set_yticklabels(range(1, len(Neigh)+1)[::-1])
# ax.set_xlim(0.5, 9.5)
# ax.set_ylim(0.5, 9.5)
# for i in range(1, len(Neigh)+1):
#     for j in range(1, len(Neigh)+1):
#         ax.text(i, 10-j, '{:.2f}'.format(dis[(i, j)]), va='center', ha='center')
# # ax.set_xticklabels(Neigh.keys())
# # ax.set_yticklabels(Neigh.keys())
# plt.savefig('similarity.pdf', format='pdf')

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.matshow(np.zeros( (len(Neigh), len(Neigh)) ), cmap='Greys')
# ax.set_xticks(range(1, len(Neigh)+1))
# ax.set_yticks(range(1, len(Neigh)+1))
# ax.set_yticklabels(range(1, len(Neigh)+1)[::-1])
# ax.set_xlim(0.5, 9.5)
# ax.set_ylim(0.5, 9.5)
# for i in range(1, len(Neigh)+1):
#     for j in range(1, len(Neigh)+1):
#         ax.text(i, 10-j, '{:.2f}'.format(1-dis[(i, j)]), va='center', ha='center')
# # ax.set_xticklabels(Neigh.keys())
# # ax.set_yticklabels(Neigh.keys())
# plt.savefig('dis-similarity.pdf', format='pdf')

model = AgglomerativeClustering(
    n_clusters=None, affinity="precomputed", distance_threshold=0, linkage="average")
model = model.fit(1-dis_matrix)

# fig = plot_dendrogram_from_sklearn(model)
# plt.ylim(-0.01, 0.90)
# ax = plt.gca()
# ax.set_xticklabels([7, 8, 9, 5, 6, 2, 4, 1, 3])
# ax.set_yticklabels([])
# plt.savefig('agglomerative-dendogram.pdf', format='pdf')

import networkx as nx
G = nx.Graph()
G.add_nodes_from(Neigh.keys())
for i in Neigh.keys():
    for j in Neigh[i]:
        G.add_edge(i, j)

from networkx import shortest_path
paths = shortest_path(G)

from networkx import edge_betweenness_centrality

edges_iteration_0 = edge_betweenness_centrality(G)
for key in edges_iteration_0.keys():
    print(key, "{:.2f}".format(edges_iteration_0[key]))
print('')

G.remove_edge(4, 5)

edges_iteration_1 = edge_betweenness_centrality(G)
for key in edges_iteration_1.keys():
    print(key, "{:.2f}".format(edges_iteration_1[key]))
print('')

G.remove_edge(4, 6)

edges_iteration_2 = edge_betweenness_centrality(G)
for key in edges_iteration_2.keys():
    print(key, "{:.2f}".format(edges_iteration_2[key]))

from networkx.algorithms import community
G.add_edge(4, 5)
G.add_edge(4, 6)
communities = list(community.girvan_newman(G))
fig = plot_dendrogram_from_networkx(communities)
plt.savefig('divisive-dendogram.pdf', format='pdf')