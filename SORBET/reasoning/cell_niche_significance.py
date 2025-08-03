"""Code for computing the IDWS score for each cell.

The computed score for each cell, c_j, is:
    Score(c_j) = \sum\limits_{i=1}^{m} ( w_p(g_i, c_j) * u(g_i) ) / ( w_p(g_i, c_j) )

Where g_i is a subgraph (m total subgraphs), u(g_i) is the label of the subgraph g_i, and w_p(g_i, c_j) is the weighted distance:
    u(g_i) is either -1 (negative phenotype) or 1 (positive phenotype)
    w_p(g_i, c_j) = d(g_i, c_j)^{-p}, where d is some distance function
"""
from typing import Optional, List, Dict, Tuple

from tqdm import tqdm as tqdm

import numpy as np
from sklearn.neighbors import kneighbors_graph
import networkx as nx

def cell_niche_significance_idw(cell_embedding: np.ndarray, subgraph_embedding: np.ndarray, subgraph_labels: np.ndarray, 
        p: Tuple[int, List[int]] = 2, distance: str = "geodesic", k: Optional[int] = 25) -> Dict[int, np.ndarray]:
    """Estimates the cell-niche significance using an inverse-distance weighting scheme over 1 or multiple scaling parameters p

    Args:
        cell_embedding: an (n x d) array representing the d-dimensional embedding of n cells. 
        subgraph_embedding: an (m x d) array representing the d-dimensional embedding of m subgraphs 
        subgraph_labels: a length m array encoding the labels (values: {0,1}) for each of the m subgraphs.
        p: chosen values for the re-scaling factors p. Passed as single integer or list of integers. 
        distance: the distance option used for computing distances between cell_embedding and subgraph_embedding.
            Options include the geodesic, euclidean and manhattan distances.
        k: an optional value for defining a k-nearest neighbors graph. Used with the 'geodesic' option for distance 
        
    Returns:
        A dictionary mapping from chosen p values to the associated embeddings.
    """
    anchor_distances = _compute_distances(cell_embedding, subgraph_embedding, distance=distance, k=k)
    
    # Convert 0/1 labeling to -1/1
    labels = np.copy(subgraph_labels)
    labels[labels < 1] = -1

    # Compute IDW weights
    if isinstance(p, int):
        p = [p]

    idw_scores = {pi:_compute_idw_score(anchor_distances, labels, pi) for pi in p}
    return idw_scores 
           
def _compute_distances(cell_embedding: np.ndarray, subgraph_embedding: np.ndarray, distance: str = "geodesic", k: Optional[int] = 25) -> np.ndarray:
    """Computes distances among cells using a specified metric. 

    Args:
        cell_embedding: an (n x d) array representing the d-dimensional embedding of n cells. 
        subgraph_embedding: an (m x d) array representing the d-dimensional embedding of m subgraphs 
        distance: the distance option used for computing distances between cell_embedding and subgraph_embedding
            Options include the geodesic, euclidean and manhattan distances.
        k: an optional value for defining a k-nearest neighbors graph. Used with the 'geodesic' option for distance 
 
    Returns:
        An (n x m) distance matrix computing the distance from n cells to the m subgraphs.
    """
    if distance == 'geodesic':
        print("Computing geodesic")
        embeddings = np.vstack([subgraph_embedding, cell_embedding])
        knn_adj_mat = kneighbors_graph(embeddings, n_neighbors=k, mode='distance', metric='minkowski', p=2)
        knn_graph = nx.from_scipy_sparse_array(knn_adj_mat)
        print("Computed KNN Graph")

        source_nodes = np.arange(subgraph_embedding.shape[0])
        print(source_nodes)

        dsts = list()
        for idx in tqdm(source_nodes):
            computed_geodesics = nx.shortest_path_length(knn_graph, source=idx, weight='weight', method='dijkstra') 
            _distances = [computed_geodesics.get(idx, np.inf) for idx in range(cell_embedding.shape[0])]
            dsts.append(_distances)

        dsts = np.array(dsts)
        print("Computed geodesic")

    elif distance == 'euclidean':
        print("Computing euclidean")
        dsts = np.linalg.norm(subgraph_embedding[:,np.newaxis,:] - cell_embedding, axis=-1)
        print("Computed Euclidean")
    elif distance == 'manhattan':
        dsts = np.sum(np.abs(subgraph_embedding[:,np.newaxis,:] - cell_embedding), axis=-1)

    return dsts

def _compute_idw_score(anchor_distances: np.ndarray, labels: np.ndarray, p: int) -> np.ndarray:
    """Computes IDW score for a given value p

    Args:
        anchor_distances: an (n x m) array encoding the distance from each of the n cells to the m subgraphs (e.g., anchor points) 
        labels: a length k array with the labels in {-1, 1} for each anchor point
        p: the factor by which each distance is weighted

    Returns:
        A length n array encoding the IDWS score (range: [-1, 1]) for each cell.
    """
    anchor_distances_inv = np.power((1 / anchor_distances), p)
    denominators = np.sum(anchor_distances_inv, axis=0)
    numerators = np.sum(anchor_distances_inv * labels[:,np.newaxis], axis=0)
    
    idw_score = numerators / denominators
    idw_score[np.isnan(idw_score)] = 0
    
    return idw_score
