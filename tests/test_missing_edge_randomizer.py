import networkx as nx
from missing_edge_randomizer.missing_edge_randomizer import (
    read_data,
    graph_cons,
    sample_graph_gen,
)
from missing_edge_randomizer.missing_edge_randomizer import sample_link_gen
from missing_edge_randomizer.missing_edge_randomizer import eigen_features
from missing_edge_randomizer.missing_edge_randomizer import community_det_greedy
from missing_edge_randomizer.missing_edge_randomizer import final


def test_read_data_binarizes_values(tmp_path):
    csv_file = tmp_path / "matrix.csv"
    csv_file.write_text("0,2,-1\n3,0,0\n")

    data = read_data(csv_file)

    assert data == [
        [0, 1, 1],
        [1, 0, 0],
    ]


def test_graph_cons_builds_bipartite_graph():
    data = [
        [1, 0],
        [0, 1],
    ]

    G, shuffled_comp_edges, edges = graph_cons(data)

    assert set(G.nodes()) == {0, 1, 2, 3}

    assert G.nodes[0]["bipartite"] == 0
    assert G.nodes[1]["bipartite"] == 0
    assert G.nodes[2]["bipartite"] == 1
    assert G.nodes[3]["bipartite"] == 1

    assert set(edges) == {(0, 2), (1, 3)}
    assert set(shuffled_comp_edges) == {(0, 3), (1, 2)}
    assert set(G.edges()) == {(0, 2), (1, 3)}

def test_sample_graph_gen_adds_edges_without_changing_original():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)

    new_G = sample_graph_gen(G, [(1, 2)])

    assert set(G.edges()) == {(0, 1)}
    assert set(new_G.edges()) == {(0, 1), (1, 2)}

def test_sample_link_gen_output_shapes():
    import random
    random.seed(0)

    G = nx.Graph()

    shuffled_comp_edges = [(0, 3), (1, 2), (1, 3), (0, 2)]
    edges = [(0, 4), (1, 5), (2, 6), (3, 7)]  # len = 4 → threshold = 2

    tmp, subsets, num_samples, box_num, box_len = sample_link_gen(
        G, shuffled_comp_edges, edges
    )

    assert num_samples == 10
    assert box_num == 2
    assert box_len == 1
    assert tmp == [0, 1, 2]

    assert len(subsets) == box_num + 1

    for i in range(box_num + 1):
        assert len(subsets[i]) == num_samples
        for sample in subsets[i]:
            assert len(sample) == tmp[i]

def test_eigen_features_counts_components_correctly():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])

    max_eigen, num_nonzero, num_components = eigen_features(G)

    assert num_components == 2
    assert num_nonzero == 2
    assert max_eigen >= 0

def test_community_det_greedy_returns_integer():
    G = nx.path_graph(4)

    result = community_det_greedy(G)

    assert isinstance(result, int)
    assert result >= 1

import numpy as np


def test_final_saves_npy_file(tmp_path):
    csv_file = tmp_path / "matrix.csv"
    csv_file.write_text("1,0\n0,1\n")

    out_prefix = tmp_path / "results"

    final(str(csv_file), str(out_prefix), "num components")

    saved_file = tmp_path / "results.npy"
    assert saved_file.exists()

    arr = np.load(saved_file, allow_pickle=True)
    assert arr.shape[1] == 10

def test_final_accepts_custom_measure_function(tmp_path):
    csv_file = tmp_path / "matrix.csv"
    csv_file.write_text("1,0\n0,1\n")

    out_prefix = tmp_path / "custom_results"

    def my_measure(G):
        return G.number_of_edges()

    final(str(csv_file), str(out_prefix), my_measure)

    saved_file = tmp_path / "custom_results.npy"
    assert saved_file.exists()

    arr = np.load(saved_file, allow_pickle=True)
    assert arr.shape[1] == 10