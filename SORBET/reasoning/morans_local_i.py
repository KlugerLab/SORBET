from typing import List, Dict, Tuple, Iterable, Any
from functools import partial

import numpy as np
import multiprocessing as mp
import pandas as pd

from tqdm import tqdm as tqdm

def calculate_weight_matrices(graph_lens: Dict[str, Dict[int, Dict[int, int]]], shell_size: int = 3, progressbar: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Computes the weight matrix w_{ij} between all pairs of nodes in graphs.

    The weight of cells i and j are:
        w_{i,j} = 1 if d(i,j) <= shell_size
        w_{i,j} = 0 if d(i,j) > shell_size

    Args:
        graph_lens: dictionary mapping each graph to a pairwise distance matrix (of the form output by networkx pairwise distances).
            Each distance matrix is encoded as a dictionary (source node) -> (target node) -> (distance) 
        shell_size: the max length to consider in weight estimation
        progressbar: display a progressbar while computing weights 

    Returns:
        Dictionary mapping graph ids to weight tuples, where each tuple includes an indexing key and assocciated weights.
    """
    _weights_dct = dict()
    for gid, lens in tqdm(graph_lens.items(), disable=(not progressbar), desc="Weight matrix computation"):
        idxing = {vid:idx for idx, vid in enumerate(lens.keys())}
        weights = np.zeros((len(idxing),) * 2)

        for vi, i in idxing.items():
            for vj, _ in filter(lambda t: t[0] != vi, filter(lambda t: t[1] <= shell_size, lens[vi].items())):
                weights[i, idxing[vj]] = 1
        
        _weights_dct[gid] = (idxing, weights)

    return _weights_dct

def _estimate_pvalues(nums, dist):
    """Helper function to estimate p-values for each subset of data. Called by calculate_local_morans_i. 
    """
    estimator_gr = lambda ni: sum(1 for di in dist if di >= ni) / len(dist)
    estimator_le = lambda ni: sum(1 for di in dist if di <= ni) / len(dist)
    return list(zip(map(estimator_gr, nums), map(estimator_le, nums)))

def calculate_local_morans_i(marker: str, expression_data: pd.DataFrame, weight_matrices: Dict[str, np.ndarray], 
        nperms=10, progressbar=False) -> Tuple[str, List[Tuple[Any]]]:
    """Computes the local Moran's i for a single instance of a marker.
    
    TODO: Include definition of Moran's local i in docstring.

    Args:
        marker: string of the marker to evaluate
        expression_data: a pandas dataframe encoding the expression data
        weight_matrices: dictionary of weight matrices computed for each subgraph 
        nperms: number of permutations to computer for each test
        progressbar: display a progressbar while computing weights 

    Returns:
        An tuple of length two. The first argument is the marker (first argument). The second argument is a list of statistics. 
        Each statistic is reported as a tuple of length 5: subgraph ID, vertex ID, local I statistic, p-value, graph label
    """
    statistics = list()
    
    for gid, (gid_idxing, wi) in tqdm(weight_matrices.items(), disable=(not progressbar)):
        N, _ = wi.shape
        W = np.sum(wi)

        graph_mask = expression_data[expression_data["Graph_ID"] == gid]

        ordered_marker_expr = np.zeros(N)
        for vj, mj in filter(lambda t: t[0] in gid_idxing, graph_mask[["Vertex_ID", marker]].values):
            ordered_marker_expr[gid_idxing[vj]] = mj

        normed_marker_expr = np.log10(ordered_marker_expr + 1)
        normed_marker_expr = normed_marker_expr - np.mean(normed_marker_expr)

        row_sums = np.sum(np.multiply(wi, normed_marker_expr), axis=1) 
        m2 = np.sum(normed_marker_expr ** 2) / (N)        

        local_is = np.multiply(normed_marker_expr, row_sums) / m2

        local_is_baseline = list()
        for _ in range(nperms):
            perm = np.random.permutation(normed_marker_expr)
            row_sums = np.sum(np.multiply(wi, perm), axis=1)
    
            local_is_baseline.extend(np.multiply(perm, row_sums) / m2)

        pvalues = _estimate_pvalues(local_is, local_is_baseline)
        
        label = graph_mask["Label"].values[0]
        vid_sort = [vid for vid, idx in sorted(gid_idxing.items(), key=lambda t: t[1])]
        
        statistics.extend((gid, vid, local_i, min(pi, pj), label) for local_i, (pi,pj), vid in zip(local_is, pvalues, vid_sort))
        
    return marker, statistics

def parallelize_moran_i_local(markers_list: List[str], expression_data: pd.DataFrame, graph_lens: Dict[str, Dict[int, Dict[int, int]]], 
        shell_size: int = 3, nperms: int = 10, nprocs: int = 16, progressbar: bool = False) -> Iterable[Tuple[str, List[Tuple[Any]]]]:
    """Parallelizes computation of Moran's loccal i over all graphs

    Args:
        markers_list: a length g list of marker labels
        expression_data: a pandas dataframe encoding the expression data
        graph_lens: dictionary of graph lengths. See calculate_weight_matrices graph_lens argument for complete description.
        shell_size: size of graph to include in each 
        nperms: number of permutations to computer for each test
        nprocs: number of parallel processes
        progressbar: display a progressbar while computing weights 

    Returns:
        An iterable of length two tuples with the first element a string and the second a tuple of statistics (see calculate_local_morans_i) 
    """
    weight_matrices = calculate_weight_matrices(graph_lens, shell_size = shell_size, progressbar = progressbar)
    
    partial_moran_i = partial(calculate_local_morans_i, expression_data = expression_data, weight_matrices = weight_matrices, nperms = nperms, progressbar = False)

    with mp.Pool(nprocs) as p:
        for marker, statistics in tqdm(p.imap_unordered(partial_moran_i, markers_list), total=len(markers_list), disable=(not progressbar), desc="Marker Computation"):
            yield marker, statistics
