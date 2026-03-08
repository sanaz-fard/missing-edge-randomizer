import csv
import random
import statistics
import numpy as np
import networkx as nx
import networkx.algorithms.community as nxcom

def read_data(path):
    with open (path, 'r') as file:
        csvreader = csv.reader(file)
        data = []
        row_counter = 0
        for row in csvreader:
            temp = []
            col_counter = 0
            for i in row:
                i = float(i)
                if i == 0:
                    temp.append(0)
                else:
                    temp.append(1)
            data.append(temp)
    return data

def graph_cons(data):
    G = nx.Graph()
    first = [i for i in range(len(data))]
    second = [i+len(data) for i in range(len(data[0]))]
    G.add_nodes_from(first, bipartite=0)
    G.add_nodes_from(second, bipartite=1)
    
    edges = []
    comp_edges = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 1:
                edges.append((i, j+len(data)))
            else:
                comp_edges.append((i, j+len(data)))
    G.add_edges_from(edges)
    shuffled_comp_edges = random.sample(comp_edges, len(comp_edges))
    return G, shuffled_comp_edges, edges

def sample_graph_gen(G, sampled_set):
    new_G = G.copy()
    new_G.add_edges_from(sampled_set)
    return new_G

def sample_link_gen(G, shuffled_comp_edges, edges):
    num_samples = 10
    threshold = int(len(edges)/2)
    box_num = 0
    box_len = 0
    if threshold <= 20:
        box_num = threshold
        box_len = 1
    else:
        box_num = 20
        box_len = int(threshold/box_num)
    tmp = [0]*(box_num+1)
    box_indx = [(box_len*i)+1 for i in range(box_num)]
    tmp [1: ] = box_indx
    subsets = [[] for i in range(box_num+1)]
    for i in range(box_num+1):
        for j in range(num_samples):
            temp = list(random.sample(shuffled_comp_edges, tmp[i]))
            subsets[i].append(temp)
    return tmp, subsets, num_samples, box_num, box_len

def centrality_measures_betweenness(new_G):
    betweenness_cent_dict = nx.betweenness_centrality(new_G, normalized=True)
    betweenness_cent_avg = statistics.variance(betweenness_cent_dict.values())
    return betweenness_cent_avg

def centrality_measures_pagerank(new_G):
    pagerank_dict = nx.pagerank(new_G, alpha=0.85, max_iter=100, tol=1e-06)
    pagerank_avg = statistics.variance(pagerank_dict.values())
    return pagerank_avg

def community_det_greedy(new_G):
    greedy_communities_list = sorted(nxcom.greedy_modularity_communities(new_G))
    greedy_community = len(greedy_communities_list)
    return greedy_community

def community_det_girvan(new_G):
    girvan_communities_list = nxcom.girvan_newman(new_G)
    girvan_communities_list_2 = next(girvan_communities_list)
    girvan_newman_community = len(girvan_communities_list_2)
    return girvan_newman_community

def community_det_louvain(new_G):
    louvain = len(nx.community.louvain_communities(new_G, weight=None, seed=123))
    return louvain

def community_det_label(new_G):
    label_propagation = len(nx.community.label_propagation_communities(new_G))
    return label_propagation

def community_det_info(new_G):
    from infomap import Infomap

    im = Infomap("--two-level --silent")

    for u, v in new_G.edges():
        im.add_link(u, v)

    im.run()

    comm = {n.node_id: n.module_id for n in im.nodes}
    info_community = len(set(comm.values()))
    return info_community

def nx_to_gt(G, bip_attr="bipartite"):
    import graph_tool.all as gt

    G_gt = gt.Graph(directed=False)

    node_list = list(G.nodes())
    vmap = {n: G_gt.add_vertex() for n in node_list}

    bip = G_gt.new_vertex_property("int")
    for n in node_list:
        bip[vmap[n]] = int(G.nodes[n][bip_attr])

    G_gt.vp["bipartite"] = bip

    for u, v in G.edges():
        G_gt.add_edge(vmap[u], vmap[v])

    return G_gt

def community_det_biSBM(new_G):
    import graph_tool.all as gt

    G_gt = nx_to_gt(new_G, bip_attr="bipartite")
    state = gt.minimize_nested_blockmodel_dl(
        G_gt,
        state_args=dict(deg_corr=True)
    )

    # Get number of blocks at each level (fine -> coarse)
    Bs = [lvl.get_B() for lvl in state.levels]

    # Find last non-trivial level before collapse to 1
    chosen_level = None
    for i in range(len(Bs) - 1):
        if Bs[i] > 1 and Bs[i + 1] == 1:
            chosen_level = i
            break

    # Fallback if no such pattern exists
    if chosen_level is None:
        # choose smallest B greater than 1
        valid = [b for b in Bs if b > 1]
        if valid:
            B_for_plot = min(valid)
        else:
            B_for_plot = 1
    else:
        B_for_plot = Bs[chosen_level]

    biSBM_community = B_for_plot
    return biSBM_community

def nx_to_igraph(G):
    import igraph as ig

    mapping = {node:i for i,node in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u,v in G.edges()]
    g = ig.Graph(edges=edges, directed=G.is_directed())
    g.vs["name"] = list(G.nodes())
    return g

def community_det_leiden(new_G):
    import leidenalg as la

    g_ig = nx_to_igraph(new_G)
    return len(la.find_partition(g_ig, la.ModularityVertexPartition))

def eigen_features(new_G):
    G_spectrum = nx.laplacian_spectrum(new_G)
    max_eigen = 0
    len_eigen = 0
    num_comp = 0
    for i in G_spectrum:
        temp = i
        if temp > max_eigen:
            max_eigen = temp
        if (np.isclose(0, temp)):
            num_comp = num_comp + 1
        else:
            len_eigen = len_eigen + 1
    return [max_eigen, len_eigen, num_comp]

def final(path, results_address, measure):
    """
    Run the missing-edge randomization pipeline.

    Parameters
    ----------
    path : str
        Path to the input CSV file.
    results_address : str
        Output path prefix for the saved NumPy file.
    measure : str or callable
        Built-in measure name or a user-defined function that takes a graph
        and returns a numeric value.
    """
    data = read_data(path)
    G, shuffled_comp_edges, edges = graph_cons(data)
    tmp, subsets, num_samples, box_num, box_len = sample_link_gen(G, shuffled_comp_edges, edges)
    measure_npy = [[0 for i in range(num_samples)] for j in range(box_num+1)]

    for i in range(box_num+1):
        for j in range(num_samples):
            samp = subsets[i][j]
            updated_G = sample_graph_gen(G, samp)

            if measure == 'betweenness':
                measure_npy[i][j] = centrality_measures_betweenness(updated_G)
            elif measure == 'pagerank':
                measure_npy[i][j] = centrality_measures_pagerank(updated_G)
            elif measure == 'greedy':
                measure_npy[i][j] = community_det_greedy(updated_G)
            elif measure == 'girvan':
                measure_npy[i][j] = community_det_girvan(updated_G)
            elif measure == 'louvain':
                measure_npy[i][j] = community_det_louvain(updated_G)
            elif measure == 'label':
                measure_npy[i][j] = community_det_label(updated_G)
            elif measure == 'infomap':
                measure_npy[i][j] = community_det_info(updated_G)
            elif measure == 'biSBM':
                measure_npy[i][j] = community_det_biSBM(updated_G)
            elif measure == 'leiden':
                measure_npy[i][j] = community_det_leiden(updated_G)
            elif measure == 'largest eigen':
                measure_npy[i][j] = eigen_features(updated_G)[0]
            elif measure == 'num eigen':
                measure_npy[i][j] = eigen_features(updated_G)[1]
            elif measure == 'num components':
                measure_npy[i][j] = eigen_features(updated_G)[2]
            elif callable(measure):
                measure_npy[i][j] = measure(updated_G)
            else:
                raise ValueError(f"Unknown measure: {measure}")

    np.save(str(results_address + '.npy'), measure_npy)