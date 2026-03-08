# missing-edge-randomizer

![Tests](https://github.com/sanaz-fard/missing-edge-randomizer/actions/workflows/tests.yml/badge.svg)

**missing-edge-randomizer** is a Python package for studying the
robustness of graph measures under missing edges. 

 The package constructs a bipartite graph from an input matrix, adds sampled missing edges to generate candidate ground-truth graphs; a set of sampled missing edges starts from smallest possible number of missing edges (no missing edge) to largest possible number of missing edges. The main functionality of package is generating these candidate ground-truth graphs and then evaluating the level of changes in the measures in this candidate ground-truth compared to original given graph. 

------------------------------------------------------------------------

## Features

-   Construct bipartite graphs from matrix-form CSV data

-   Randomly sample missing edges

-   Generate perturbed graph ensembles

-   Evaluate structural graph measures, including:

    ### Centrality

    -   Betweenness centrality variance
    -   PageRank variance

    ### Community Detection

    -   Greedy modularity
    -   Girvan-Newman
    -   Louvain
    -   Label propagation
    -   Infomap
    -   Bipartite stochastic block model (biSBM)
    -   Leiden

    ### Spectral Properties

    -   Largest Laplacian eigenvalue
    -   Number of nonzero Laplacian eigenvalues
    -   Number of connected components

-   Supports user-defined custom graph measures

------------------------------------------------------------------------

## Installation

### Basic installation

``` bash
pip install .
```

### Installation with advanced community-detection features

``` bash
pip install .[advanced]
```

The advanced option installs optional dependencies required for Infomap,
Leiden, igraph, and graph-tool based methods.

------------------------------------------------------------------------

## Usage

### Using a built-in measure

``` python
from missing_edge_randomizer import final

final("data.csv", "results", "num components")
```

### Using a custom user-defined measure

``` python
from missing_edge_randomizer import final

def my_measure(G):
    return G.number_of_edges()

final("data.csv", "custom_results", my_measure)
```

------------------------------------------------------------------------

## Input Format

The package expects a CSV file representing a matrix.

Each entry is interpreted as:

-   `0` → no edge\
-   nonzero value → edge present

The matrix is treated as the biadjacency matrix of a bipartite graph.

------------------------------------------------------------------------

## Output

Results are saved as a NumPy `.npy` file containing the evaluated
measure across sampled graphs.

------------------------------------------------------------------------

## Example Workflow

1.  Prepare a CSV matrix describing bipartite connections.
2.  Choose a structural measure (built-in or custom).
3.  Run the `final()` pipeline.
4.  Analyze the saved NumPy results.

------------------------------------------------------------------------

## Testing

To run the test suite:

``` bash
pytest
```

------------------------------------------------------------------------

## Citation

If you use this software in academic research, please cite the
associated paper: "A Technique for Assessing the Robustness of Network
Structural Metrics under Incomplete Data".

------------------------------------------------------------------------

## Author

Sanaz Hasanzadeh Fard and Emily Dolson

------------------------------------------------------------------------

## License

MIT License
