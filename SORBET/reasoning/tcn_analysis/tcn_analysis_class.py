from typing import List, Dict, Tuple
import numpy as np
from scipy.cluster import hierarchy as hier
from scipy.spatial import distance as ssd
from scipy.stats import ks_2samp, fisher_exact, mannwhitneyu, probplot, f, combine_pvalues, norm
from statsmodels.discrete.discrete_model import NegativeBinomial
import SORBET.data_handling as data_handling
from .tcn_class import TCN
from functools import reduce
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, homogeneity_score
import itertools
import copy
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import networkx as nx
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool, cpu_count
import statsmodels.api as sm
import os
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

class TCNAnalysis:
    """
    A class to manage the overall TCN analysis process.
    """
    @staticmethod
    def calculate_TCNs(cell_type_mapping: Dict[Tuple[int, int], Dict[int, str]], cell_type_indexing: Dict[str, int]) -> List[TCN]:
        """
        Calculates TCN arrays for every vertex in the omics graph per run and field of view.

        Args:
            cell_type_mapping: Mapping from (run_id, fov_id) to vertex->cell type name.
            cell_type_indexing: Mapping from cell type name to column index in TCN arrays.

        Returns:
            List[TCN]: A list of TCN objects, one per vertex per FOV.

        Notes:
            If a graph file is missing, that (run_id, fov_id) pair is skipped.
        """
        tcns = []
        for (run_id, fov_id), _ in cell_type_mapping.items():
            # Question: can we assume this graph_fpath? If not should change to a parameter with a default value 
            graph_fpath = f'data/CosMx/graphs_py/complete/run_{run_id}_fov_{fov_id}.p'
            try:
                graph = data_handling.load_omicsgraph(graph_fpath)
            except FileNotFoundError:
                print(f'No modeled graph found for run {run_id} and fov {fov_id}')
                continue
            
            n_cts = len(cell_type_indexing)

            def map_cell_type_index(vertex):
                cell_type = cell_type_mapping[(run_id, fov_id)].get(vertex, None)
                return cell_type_indexing.get(cell_type, None)
            
            for vi in graph.vertices:
                tcn_arr = np.zeros((4, n_cts))
                vi_ct_index = map_cell_type_index(vi)
                if vi_ct_index is not None:
                    tcn_arr[0, vi_ct_index] += 1
                seen = set([vi])
                for k in range(1, 4):
                    neighborhood_k = set(graph.get_khop_neighborhood(vi, k)).difference(seen)
                    neighborhood_k = neighborhood_k.difference(seen)
                    for vj in neighborhood_k:
                        ctj_idx = map_cell_type_index(vj)
                        if ctj_idx is not None:
                            tcn_arr[k, ctj_idx] += 1
                    seen.update(neighborhood_k)
                tcns.append(TCN([run_id], [fov_id], tcn_arr, [graph.graph_label], [vi], cell_type_indexing))            
        
        return tcns
    

    @staticmethod
    def calculate_TCNs_with_IDWS(cell_type_mapping: Dict[Tuple[int, int], Dict[int, str]],
                                 cell_type_indexing: Dict[str, int],
                                 subgraphs: np.ndarray,
                                 scores: np.ndarray,
                                 vertices: np.ndarray,
                                 neighberhood_size: int = 3,
                                 add_markers: bool = False) -> Tuple[List['TCN'], Dict[Tuple[int, int], Dict[int, float]], Dict[Tuple[int, int], Dict[int, np.ndarray]]]:
        """
        Calculates TCN arrays and IDWS scores for vertices in modeled subgraphs.

        Args:
            cell_type_mapping: Mapping from (run_id, fov_id) to vertex->cell type name.
            cell_type_indexing: Mapping from cell type name to column index.
            subgraphs: Array of subgraph identifiers aligned with `vertices`.
            scores: Array of IDWS scores aligned with `vertices`.
            vertices: Array of vertex identifiers aligned with `subgraphs`.
            neighberhood_size: Number of hops to include (default 3).
            add_markers: If True, include marker-level TCN arrays.

        Returns:
            tcns: List[TCN] — TCN objects with optional marker arrays.
            scores_per_fov: Dict[(run_id, fov_id), Dict[cell_id, float]] — IDWS scores per cell.
            cell_profiles: Dict[(run_id, fov_id), Dict[cell_id, np.ndarray]] — marker profiles per cell.

        Notes:
            Each vertex is processed only once per FOV; missing graph files are skipped.
        """
        tcns = []
        scores_per_fov = defaultdict(dict)
        cell_profiles = defaultdict(lambda: defaultdict(lambda: None))
        re_key = r'run\_(\d+)\_fov\_(\d+)\_sg\_(\d+)'
        per_fov = defaultdict(set)

        # Group subgraphs by FOV
        for k in np.unique(subgraphs):
            m = re.match(re_key, str(k))
            if m:
                run, fov = int(m.group(1)), int(m.group(2))
                per_fov[(run, fov)].add(k)
        
        # Process each FOV
        for (run_id, fov_id), subgraph_keys in per_fov.items():
            graph_fpath = f'data/CosMx/graphs_py/complete/run_{run_id}_fov_{fov_id}.p'
            try:
                graph = data_handling.load_omicsgraph(graph_fpath)
            except FileNotFoundError:
                print(f'No modeled graph found for run {run_id} and fov {fov_id}')
                continue

            n_cts = len(cell_type_indexing)
            processed_vertices = set()
            # Calculate TCN and IDWS for each modeled vertex
            for sgi in subgraph_keys:
                subgraph_mask = subgraphs == sgi
                vertices_sg = vertices[subgraph_mask]
                chosen_scores = scores[subgraph_mask]

                for vi, score in zip(vertices_sg, chosen_scores):
                    if vi not in processed_vertices:
                        processed_vertices.add(vi)
                        tcn_arr = np.zeros((1 + neighberhood_size, n_cts))
                        marker_tcn_arr = np.zeros((4, len(graph.markers))) if add_markers else None

                        vi_ct_index = cell_type_mapping[(run_id, fov_id)].get(vi, None)
                        if vi_ct_index is None:
                            continue
                        cell_type_indices = {key: {k: [] for k in range(0, 1 + neighberhood_size)} for key in cell_type_indexing}
                        if vi_ct_index is not None:
                            tcn_arr[0, cell_type_indexing[vi_ct_index]] += 1
                            cell_type_indices[vi_ct_index][0].append(vi)
                        
                        if add_markers:
                            node_attr = list(graph.node_attributes[vi].values())
                            marker_tcn_arr[0, :] = np.array(node_attr)
                            cell_profiles[(run_id, fov_id)][vi] = np.array(node_attr)

                        seen = set([vi])
                        for k in range(1, 1 + neighberhood_size):
                            neighborhood_k = set(graph.get_khop_neighborhood(vi, k)).difference(seen)
                            for vj in neighborhood_k:
                                ctj_idx = cell_type_mapping[(run_id, fov_id)].get(vj, None)
                                if ctj_idx is not None:
                                    tcn_arr[k, cell_type_indexing[ctj_idx]] += 1
                                    cell_type_indices[ctj_idx][k].append(vj) 
                                    if add_markers and vj in graph.node_attributes:
                                        node_attr = list(graph.node_attributes[vj].values())
                                        marker_tcn_arr[k, :] += np.array(node_attr)
                            seen.update(neighborhood_k)

                        tcns.append(TCN([run_id], [fov_id], tcn_arr, [graph.graph_label], [vi], cell_type_indexing, marker_tcn_arr=marker_tcn_arr, cell_type_indeces_dict=cell_type_indices))
                        scores_per_fov[(run_id, fov_id)][vi] = score
            
        return tcns, scores_per_fov, cell_profiles
    

    def compute_hierarchical_clustering(tcns, method='average', cell_type_group_to_remove=None, optimal_ordering=False):
        """
        Performs hierarchical clustering on flattened TCN representations and selects the optimal number of clusters.

        Args:
            tcns: List[TCN] — the list of TCN objects to cluster.
            method: str — linkage method for clustering (default 'average').
            cell_type_group_to_remove: List of cell type names to exclude before clustering.
            optimal_ordering: bool — enable optimal leaf ordering if True.

        Returns:
            clusters: List[TCN] — one summed TCN per cluster.
            cluster_labels: np.ndarray — cluster label per input TCN.

        Raises:
            ValueError: if `tcns` is empty.
        """
        if not tcns:
            raise ValueError("Cannot cluster empty TCN list.")
        # Prepare the data array for clustering
        if cell_type_group_to_remove is not None:
            cell_types_to_keep_indices = [i for i, ct in enumerate(tcns[0].cell_type_indexing) if ct not in cell_type_group_to_remove]
            tcn_arrs = [tcn.get_normed_representation()[:, cell_types_to_keep_indices] for tcn in tcns]
        else:
            tcn_arrs = [tcn.get_normed_representation() for tcn in tcns]

        tcn_arrs = np.concatenate([tcn_arr.reshape(1, -1) for tcn_arr in tcn_arrs], axis=0)
        tcn_labels = [tcn.label[0] for tcn in tcns]  # Assuming label is a property of each TCN

        # Compute distance matrix and linkage
        dmatrix = ssd.pdist(tcn_arrs, metric='correlation')
        dmatrix[np.isnan(dmatrix)] = 0
        Z = hier.linkage(dmatrix, method=method, optimal_ordering=optimal_ordering)

        # Evaluate cluster configurations
        range_n_clusters = list(range(10, 1000, 10))  # Searching for the best number of clusters
        best_score = -1
        alpha = 0.9  # Weight between homogeneity and silhouette

        for n_clusters in range_n_clusters:
            cluster_labels = hier.fcluster(Z, n_clusters, criterion='maxclust')
            homogeneity = homogeneity_score(tcn_labels, cluster_labels)
            silhouette = silhouette_score(tcn_arrs, cluster_labels)

            # Composite score
            score = alpha * homogeneity + (1 - alpha) * silhouette
            if score > best_score:
                best_score = score
                optimal_clusters = n_clusters
                optimal_cluster_labels = cluster_labels
                print(f'Optimal number of clusters: {optimal_clusters}, Homogeneity: {homogeneity}, Silhouette: {silhouette}, Composite: {score}')

        # Aggregate TCNs based on optimal cluster assignment
        clusters = []
        for cluster_label in np.unique(optimal_cluster_labels):
            mask = optimal_cluster_labels == cluster_label
            tcns2sum = [tcn for tcn, m in zip(tcns, mask) if m]
            sum_tcn = TCNAnalysis.sum_tcn(tcns2sum)
            sum_tcn.set_avarage_neighborhood_sizeS(sum_tcn.get_avarage_neighborhood_sizes())  # Adjust average neighborhood sizes
            clusters.append(sum_tcn)

        return clusters, optimal_cluster_labels



    @staticmethod
    def filter_significant_clusters(cluster_tcns: List[TCN], idws_scores: Dict[Tuple[int, int], Dict[int, float]], min_cells: int = 500, min_patients: int = 2, significance_level: float = 0.05, use_original_labels: bool = False) -> Tuple[List[TCN], List[float], List[int]]:
        """
        Filters clusters by comparing their IDWS score distributions against all others using the KS test.

        Args:
            cluster_tcns: List[TCN] — one TCN per cluster.
            idws_scores: Mapping from (run_id,fov_id) to cell->score.
            min_cells: Minimum center cells in a cluster (default 500).
            min_patients: Minimum unique patients (default 2).
            significance_level: p-value threshold (default 0.05).
            use_original_labels: If True, compare original labels instead.

        Returns:
            significant_tcns: List[TCN] — clusters passing all filters.
            p_values: List[float] — KS test p-values per cluster.
            indices: List[int] — indices of significant clusters.
        """
        if use_original_labels:
            clusters_scores_dict = {c_id: cluster.label for c_id, cluster in enumerate(cluster_tcns)}
        else:
            clusters_scores_dict = {c_id: [idws_scores[(run_id, fov_id)][cell_id] for run_id, fov_id, cell_id in zip(cluster.run_id, cluster.fov_id, cluster.center_cell_index)] for c_id, cluster in enumerate(cluster_tcns)}

        p_values = []
        for i, cluster in enumerate(cluster_tcns):
            cluster_scores = clusters_scores_dict[i]
            other_clusters_scores = np.concatenate([clusters_scores_dict[j] for j in range(len(cluster_tcns)) if j != i])
            res = ks_2samp(cluster_scores, other_clusters_scores)
            p_values.append(res.pvalue)
        significanct_clusters = [cluster for cluster, p_value in zip(cluster_tcns, p_values) if p_value < significance_level and len(cluster.center_cell_index) >= min_cells and len(set(cluster.label)) >= min_patients]
        significant_p_values = [p_value for cluster, p_value in zip(cluster_tcns, p_values) if p_value < significance_level and len(cluster.center_cell_index) >= min_cells and len(set(cluster.label)) >= min_patients]
        significant_clusters_indices = [i for i, clust_p_val_tuple in enumerate(zip(cluster_tcns, p_values)) if clust_p_val_tuple[1] < significance_level and len(clust_p_val_tuple[0].center_cell_index) >= min_cells and len(set(clust_p_val_tuple[0].label)) >= min_patients]
        return significanct_clusters, significant_p_values, significant_clusters_indices
    
    @staticmethod
    def compute_original_vs_idws_p_values(cluster_mask: np.ndarray, labels: np.ndarray, thresholded_idws: np.ndarray, significant_clusters_inds: None | List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Fisher exact‐test p-values comparing original labels vs. IDWS thresholds per cluster.

        Args:
            cluster_mask: 1D array of cluster assignments per cell.
            labels: 1D array of original binary labels per cell.
            thresholded_idws: 1D array of −1/0/+1 per cell from thresholded IDWS scores.
            significant_clusters_inds: Optional list of cluster IDs to include; if None, use all.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - p_values_labels: p-values comparing original labels inside vs. outside each cluster.
                - p_values_idws: p-values comparing IDWS classes inside vs. outside.
        """
        if significant_clusters_inds is None:
            unique_clusters = np.unique(cluster_mask)
        else:
            unique_clusters = significant_clusters_inds
        p_values_labels = []
        p_values_idws = []

        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_mask == cluster)[0]

            # Contingency table for original labels
            contingency_table_labels = np.array([
                [np.sum((labels[cluster_indices] == 0)), np.sum((labels[cluster_indices] == 1))],
                [np.sum(labels == 0) - np.sum((labels[cluster_indices] == 0)), np.sum(labels == 1) - np.sum((labels[cluster_indices] == 1))]
            ])
            _, p_value_labels = fisher_exact(contingency_table_labels)
            p_values_labels.append(p_value_labels)

            # Contingency table for thresholded IDWS scores
            contingency_table_idws = np.array([
                [np.sum((thresholded_idws[cluster_indices] == -1)), np.sum((thresholded_idws[cluster_indices] == 1))],
                [np.sum(thresholded_idws == -1) - np.sum((thresholded_idws[cluster_indices] == -1)), np.sum(thresholded_idws == 1) - np.sum((thresholded_idws[cluster_indices] == 1))]
            ])
            _, p_value_idws = fisher_exact(contingency_table_idws)
            p_values_idws.append(p_value_idws)

        return np.array(p_values_labels), np.array(p_values_idws)

    @staticmethod
    def plot_p_values_comparison(p_values_labels: np.ndarray, p_values_idws: np.ndarray, significant_cluster_numbers: List[int]) -> None:
        """
        Plots a scatter of original‐vs‐IDWS p-values for significant clusters.

        Args:
            p_values_labels: Array of p-values from original-label tests.
            p_values_idws: Array of p-values from IDWS-based tests.
            significant_cluster_numbers: List of cluster IDs to annotate.

        Returns:
            None — displays a matplotlib scatter with a y=x reference line.
        """
        plt.figure(figsize=(10, 10))
        plt.scatter(p_values_labels, p_values_idws, alpha=0.5)

        for i, cluster_num in enumerate(significant_cluster_numbers):
            plt.text(p_values_labels[i], p_values_idws[i], str(cluster_num), fontsize=9, ha='right')

        plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
        plt.xlabel('P-values (Original Labels)')
        plt.ylabel('P-values (Thresholded IDWS Scores)')
        # plt.title('Comparison of P-values')
        plt.grid(True)
        plt.show()

    @staticmethod
    def filter_tcn_list_according_to_center_cell(tcn_list: List[TCN], cell_type_group: List[str]) -> Tuple[List[TCN], List[bool]]:
        """
        Keeps only those TCNs whose center cell belongs to one of the given types.

        Args:
            tcn_list: List of TCN objects.
            cell_type_group: Cell types to allow at the center.

        Returns:
            Tuple:
                - filtered_tcns: the subset of TCNs passing the filter.
                - mask: list of booleans same length as `tcn_list`.
        """
        tcn_group_mask = [tcn.is_tcn_center_cell_in_types(cell_type_group) for tcn in tcn_list]
        tcn_group = [tcn for tcn, mask in zip(tcn_list, tcn_group_mask) if mask]
        return tcn_group, tcn_group_mask
    
    @staticmethod
    def filter_tcn_list_homogenous_clusters(tcn_list: List[TCN], cell_type_group: List[str]) -> List[TCN]:
        """
        Keeps only those TCNs composed exclusively of the given cell types.

        Args:
            tcn_list: List of TCN objects.
            cell_type_group: Cell types that may appear anywhere in the TCN.

        Returns:
            List[TCN]: subset that are homogeneous in the given types.
        """
        tcn_group_mask = [tcn.is_homogenous_tcn_in_type(cell_type_group) for tcn in tcn_list]
        tcn_group = [tcn for tcn, mask in zip(tcn_list, tcn_group_mask) if mask]
        return tcn_group
    
    @staticmethod
    def sum_tcn(tcns: List[TCN]) -> TCN:
        """
        Element‐wise sums a list of TCN objects via their `__add__`.

        Args:
            tcns: List of TCN to sum.

        Returns:
            TCN: the cumulative sum.
        """
        return reduce(lambda x, y: x + y, tcns)
    
    @staticmethod
    def map_tcn_list_to_meta_types(tcn_list: List[TCN], meta_types: Dict[str, str]) -> List[TCN]:
        """
        Applies `get_remap_cell_types_tcn_object` to every TCN in a list. Basicly allows to remap cell types to new types. 

        Args:
            tcn_list: List of TCN objects.
            meta_types: Mapping from original cell type → meta‐group name or None.

        Returns:
            List[TCN]: remapped clones.
        """
        ...
        new_tcn_list = []
        for tcn in tcn_list:
            new_tcn = tcn.get_remap_cell_types_tcn_object(meta_types)
            new_tcn_list.append(new_tcn)
        return new_tcn_list
    
    @staticmethod
    def compute_pca_for_tcn_list(tcn_list: List[TCN], use_center_cells: bool = False, pca_n_comp: int = 10) -> np.ndarray:
        """
        Runs PCA on either each center‐cell embedding or the full normalized TCNs.

        Args:
            tcn_list: List of TCN objects.
            use_center_cells: If True, uses only the hop-0 row per TCN.
            pca_n_comp: Number of principal components to return.

        Returns:
            np.ndarray: shape (n_tcns, pca_n_comp) of PCA scores.
        """
        if use_center_cells:
            tcn_arrs = [tcn.tcn_arr[0,:] for tcn in tcn_list]
        else:
            sum_of_all_tcns = TCNAnalysis.sum_tcn(tcn_list)
            avarage_nei_sizes = sum_of_all_tcns.get_avarage_neighborhood_sizes()
            tcn_arrs = [tcn.get_normed_representation() / avarage_nei_sizes.reshape(-1, 1) for tcn in tcn_list]
        tcn_arrs = np.concatenate([tcn_arr.reshape(1, -1) for tcn_arr in tcn_arrs], axis=0)
        # compute pca and then tsne
        pca = PCA(n_components=pca_n_comp)
        pca_result = pca.fit_transform(tcn_arrs)
        return pca_result
    
    @staticmethod
    def calculate_original_marker_rep_TCN_with_IDWS(subgraphs: np.ndarray, scores: np.ndarray, vertices: np.ndarray) -> Tuple[List[TCN], Dict[Tuple[int, int], Dict[int, float]], List[int]]:
        """
        Builds marker‐based TCNs (4 hops × markers) with associated IDWS scores.

        Args:
            subgraphs: 1D array of subgraph IDs per vertex.
            scores: 1D array of IDWS scores per vertex.
            vertices: 1D array of vertex IDs aligned with `subgraphs` & `scores`.

        Returns:
            tcns: List of marker‐based TCNs.
            scores_per_fov: per‐(run,fov)→cell→score dict.
            tcns_center_cell_indices: List of original vertex‐array indices.
        """
        tcns = []
        scores_per_fov = defaultdict(dict) 
        tcns_center_cell_indices = []
        vertices_indices = np.arange(len(vertices))
        re_key = r'run\_(\d+)\_fov\_(\d+)\_sg\_(\d+)'
        per_fov = defaultdict(set)

        # Group subgraphs by FOV
        for k in np.unique(subgraphs):
            m = re.match(re_key, str(k))
            if m:
                run, fov = int(m.group(1)), int(m.group(2))
                per_fov[(run, fov)].add(k)
        
        # Process each FOV
        for (run_id, fov_id), subgraph_keys in per_fov.items():
            graph_fpath = f'data/CosMx/graphs_py/complete/run_{run_id}_fov_{fov_id}.p'
            try:
                graph = data_handling.load_omicsgraph(graph_fpath)
            except FileNotFoundError:
                print(f'No modeled graph found for run {run_id} and fov {fov_id}')
                continue
            
            processed_vertices = set()
            # Calculate TCN and IDWS for each modeled vertex
            for sgi in subgraph_keys:
                subgraph_mask = subgraphs == sgi
                vertices_sg = vertices[subgraph_mask]
                chosen_scores = scores[subgraph_mask]
                sg_vertices_indices = vertices_indices[subgraph_mask]

                for vi, score, vi_idx in zip(vertices_sg, chosen_scores, sg_vertices_indices):
                    if vi not in processed_vertices:
                        processed_vertices.add(vi)
                        # tcn in this case is of size 4 x N where N is the number of original markers
                        tcn_arr = np.zeros((4, len(graph.markers)))
                        # the first row is the center cell marker distribution
                        node_attr = list(graph.node_attributes[vi].values())
                        tcn_arr[0, :] = np.array(node_attr)
                        # the rest are the sum of the marker distribution of the neighborhood
                        seen = set([vi])
                        for k in range(1, 4):
                            neighborhood_k = set(graph.get_khop_neighborhood(vi, k)).difference(seen)
                            for vj in neighborhood_k:
                                node_attr = list(graph.node_attributes[vj].values())
                                tcn_arr[k, :] += np.array(node_attr)
                            seen.update(neighborhood_k)
                            
                        tcns.append(TCN([run_id], [fov_id], tcn_arr, [graph.graph_label], [vi], {marker: i for i, marker in enumerate(graph.markers)}))
                        scores_per_fov[(run_id, fov_id)][vi] = score
                        tcns_center_cell_indices.append(vi_idx)
        return tcns, scores_per_fov, tcns_center_cell_indices
    

    @staticmethod
    def _generate_combinations(cell_type_group: List[str], cell_type_indexing: Dict[str, int], is_merge_2_and_3=True) -> List[Tuple[np.ndarray, List[int]]]:
        """
        Generates all binary presence/absence combinations for given cell types over hops.

        Args:
            cell_type_group: List of cell types to include.
            cell_type_indexing: Map from cell type name to column index.
            is_merge_2_and_3: If True, merge hops 2 and 3 for combination generation.

        Returns:
            List of tuples (comb_matrix, indices) where:
                comb_matrix: np.ndarray of shape (4 or 3, len(indices)), binary patterns.
                indices: list of column indices corresponding to `cell_type_group`.
        """
        indices = [cell_type_indexing[cell] for cell in cell_type_group]
        if is_merge_2_and_3:
            combinations = list(itertools.product([0, 1], repeat=2 * len(indices)))
            combinations = [np.vstack([np.zeros((1, len(indices)), dtype=int), np.array(comb).reshape((2, len(indices))), np.zeros((1, len(indices)))]) for comb in combinations]
        else:
            combinations = list(itertools.product([0, 1], repeat=3 * len(indices)))
            combinations = [np.vstack([np.zeros((1, len(indices)), dtype=int), np.array(comb).reshape((3, len(indices)))]) for comb in combinations]
        return [(comb, indices) for comb in combinations]

    @staticmethod
    def _match_combinations(args):
        """
        Helper for multiprocessing; matches TCN data against provided combinations.

        Args:
            args: A tuple containing:
                - tcn_data: dict with keys 'binary_tcn' and 'tcn'.
                - combinations: list of (comb_matrix, indices).
                - indices: list of indices to check in binary_tcn.
                - is_merge_2_and_3: merge flag as above.

        Returns:
            List of matches where combination pattern equals binary_tcn slices.
        """
        tcn_data, combinations, indices, is_merge_2_and_3 = args
        results = []
        binary_tcn = np.array(tcn_data['binary_tcn'])
        for combination, _ in combinations:
            if is_merge_2_and_3:
                match = np.all(binary_tcn[1, indices] == combination[1, :]) and \
                        (np.all(binary_tcn[2, indices] == combination[2, :]) or np.all(binary_tcn[3, indices] == combination[2, :]))
            else:
                match = np.all(binary_tcn[:, indices] == combination)
            if match:
                results.append((combination, tcn_data['tcn']))
        return results

    @staticmethod
    def count_tcns_and_patients(tcns: List[TCN], cell_type_group: List[str], is_merge_2_and_3=True, batch_size=100) -> Dict[Tuple[Tuple[int], int], List[TCN]]:
        """
        Batches through all presence/absence patterns and groups TCNs accordingly.

        Args:
            tcns: list of TCN objects to count.
            cell_type_group: Cell types to consider in combinations.
            is_merge_2_and_3: Whether to merge hops 2 and 3.
            batch_size: number of combinations per parallel batch.

        Returns:
            Dict mapping each flattened combination tuple to list of TCNs matching it.
        """
        results = {}
        cell_type_indexing = tcns[0].cell_type_indexing
        combinations = TCNAnalysis._generate_combinations(cell_type_group, cell_type_indexing, is_merge_2_and_3)
        
        tcn_data_list = [{'binary_tcn': tcn.get_binary_representation(), 'tcn': tcn} for tcn in tcns]

        # Create batches of combinations
        combination_batches = [combinations[i:i + batch_size] for i in range(0, len(combinations), batch_size)]
        args_list = [(tcn_data, batch, batch[0][1], is_merge_2_and_3) for tcn_data in tcn_data_list for batch in combination_batches]
        
        with Pool(cpu_count()) as pool:
            batch_results = pool.map(TCNAnalysis._match_combinations, args_list)

        # Combine results from all batches
        for batch_result in batch_results:
            for combination, tcn in batch_result:
                key = tuple(combination.flatten())
                if key not in results:
                    results[key] = [tcn]
                else:
                    results[key].append(tcn)

        return results
    
    @staticmethod
    def plot_combination_heatmap(combination: np.ndarray, cell_type_group: List[str], remove_center: bool = True, combination_number: int = 1) -> None:
        """
        Saves a heatmap SVG for a single binary combination of cell types.

        Args:
            combination: Flattened binary combination array.
            cell_type_group: List of cell type names.
            remove_center: If True, omit hop-0 row from plot.
            combination_number: Identifier used in SVG filename.

        Returns:
            None — writes 'combination_heatmap_{combination_number}.svg'.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Convert the combination to the appropriate 4 * len(indices) matrix
        len_indices = len(cell_type_group)
        tcn = np.array(combination).reshape((4, len_indices))
        if remove_center:
            tcn = tcn[1:]

        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"], N=256)
        
        # Plot the heatmap
        ax.imshow(tcn, cmap='Greys')
        ax.set_xticks(np.arange(tcn.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(tcn.shape[0] + 1) - .5, minor=True)
        ax.tick_params(which='minor', size=0)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.set_xticks(np.arange(tcn.shape[1]))
        ax.set_yticks(np.arange(tcn.shape[0]))
        ax.set_xticklabels(cell_type_group, fontsize=18)
        ax.set_yticklabels(['Hop 1', 'Hop 2', 'Hop 3'] if remove_center else ['Center', 'Hop 1', 'Hop 2', 'Hop 3'], fontsize=18)
        plt.savefig(f"combination_heatmap_{combination_number}.svg", dpi=1200, facecolor='white', edgecolor='white', bbox_inches='tight')
        plt.close()
        # plt.show()

    @staticmethod
    def filter_significant_combinations(tcns_dict: Dict[Tuple[int, ...], List['TCN']], 
                                        idws_scores: Dict[Tuple[int, int], Dict[int, float]], 
                                        min_cells: int = 500, 
                                        min_patients: int = 2, 
                                        significance_level: float = 0.05, 
                                        use_original_labels: bool = False) -> Tuple[Dict[Tuple[int, ...], List['TCN']], Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], int]]:
        """
        Filters TCN groups by Fisher's test on inside vs. outside responder counts.

        Args:
            tcns_dict: Map from combination tuple to list of TCNs.
            idws_scores: as above.
            min_cells: Min TCN count to consider.
            min_patients: Min unique patients required.
            significance_level: p-value cutoff.
            use_original_labels: If True, use TCN.label instead of idws_scores.

        Returns:
            filtered_dict: combinations meeting criteria.
            p_values: p-value per combination.
            significant_indices: mapping from running index to combination.
        """
        filtered_tcns_dict = {comb: tcns for comb, tcns in tcns_dict.items() if len(tcns) >= min_cells}
        
        significant_combinations = {}
        significant_p_values = {}
        significant_indices = {}

        count = 0  # Initialize count for tracking significant combinations

        # Precompute global counts for all TCNs for "outside" distribution
        global_positive_count, global_negative_count = TCNAnalysis.get_global_center_cell_dist(tcns_dict, idws_scores, use_original_labels)
        print(f"Global counts for positive {global_positive_count}, Global counts for negative {global_negative_count}")

        for combination, cluster_tcns in filtered_tcns_dict.items():
            # Initialize counts for the "inside" distribution of the current TCN cluster
            inside_positive_count, inside_negative_count = 0, 0

            inside_positive_count, inside_negative_count = TCNAnalysis.get_inside_center_cell_dist(idws_scores, use_original_labels, inside_positive_count, inside_negative_count, cluster_tcns)
            # print(f"local counts for positive {inside_positive_count}, local counts for negative {inside_negative_count}")

            # Compute the "outside" distribution by subtracting "inside" counts from global counts
            outside_positive_count = global_positive_count - inside_positive_count
            outside_negative_count = global_negative_count - inside_negative_count

            # Construct the contingency table with rows for "inside" and "outside" distributions
            contingency_table = np.array([
                [inside_positive_count, inside_negative_count],   # Inside TCN cluster
                [outside_positive_count, outside_negative_count]   # Outside TCN cluster (all other TCNs)
            ])

            # Apply Fisher's exact test
            _, p_value = fisher_exact(contingency_table)

            # Check if the result is significant
            if p_value < significance_level:
                significant_combinations[combination] = cluster_tcns
                significant_p_values[combination] = p_value
                significant_indices[count] = combination
                count += 1  # Increment count for each significant combination

        return significant_combinations, significant_p_values, significant_indices

    @staticmethod
    def get_global_center_cell_dist(tcns_dict, idws_scores, use_original_labels):
        """
        Computes total global counts of positive and negative center cells.

        Args:
            tcns_dict: as above.
            idws_scores: as above.
            use_original_labels: If True, count by TCN.label.

        Returns:
            Tuple (global_positive_count, global_negative_count).
        """
        global_positive_count, global_negative_count = 0, 0
        for _, cluster_tcns in tcns_dict.items():
            global_positive_count, global_negative_count = TCNAnalysis.get_inside_center_cell_dist(idws_scores, use_original_labels, global_positive_count, global_negative_count, cluster_tcns)
        return global_positive_count,global_negative_count

    @staticmethod
    def get_inside_center_cell_dist(idws_scores, use_original_labels, global_positive_count, global_negative_count, cluster_tcns):
        """
        Accumulates inside-cluster responder/non-responder counts.

        Args:
            idws_scores: as above.
            use_original_labels: If True, use labels.
            pos_count: initial positive count.
            neg_count: initial negative count.
            cluster_tcns: list of TCNs in the cluster.

        Returns:
            Updated (pos_count, neg_count) after processing cluster.
        """
        for tcn in cluster_tcns:
            if use_original_labels:
                if tcn.label[0] == 1:
                    global_positive_count += 1
                else:
                    global_negative_count += 1
            else:
                for run_id, fov_id, cell_id in zip(tcn.run_id, tcn.fov_id, tcn.center_cell_index):
                    if (run_id, fov_id) in idws_scores and cell_id in idws_scores[(run_id, fov_id)]:
                        score = idws_scores[(run_id, fov_id)][cell_id]
                        if score > 0.5:
                            global_positive_count += 1
                        elif score < -0.5:
                            global_negative_count += 1
        return global_positive_count, global_negative_count


    @staticmethod
    def validate_TCN(tcn: 'TCN', combination: np.ndarray) -> bool:
        """
        Verifies a TCN's binary and index maps against expected combination.

        Args:
            tcn: TCN object with `tcn_arr` and `cell_type_indeces_dict`.
            combination: Flattened binary pattern array for relevant cell types.

        Returns:
            True if actual matches expected, False otherwise (prints discrepancy).
        """
        cell_type_group = ["T.CD4", "T.CD8", "T.cell", "B.cell", "Macrophage", "NK"] 
        num_hops = 4  # There are 4 hops including the center (hop 0)
        num_relevant_cell_types = len(cell_type_group)
        combination = np.array(combination)

        # Reshape the combination array to match the hop/cell type structure (4 hops x relevant cell types)
        expected_pattern = combination.reshape((num_hops, num_relevant_cell_types))

        # Step 1: Check the TCN array (tcn_arr) against the expected pattern for the relevant cell types
        for hop in range(num_hops):
            for cell_type_idx, cell_type in enumerate(cell_type_group):
                expected_cells = expected_pattern[hop, cell_type_idx]
                # Get the index for the relevant cell type from the TCN's indexing
                actual_cells = tcn.tcn_arr[hop, tcn.cell_type_indexing[cell_type]] if cell_type in tcn.cell_type_indexing else 0

                if expected_cells > 0 and actual_cells == 0:
                    # Expected cells but none found in the TCN array
                    print(f"Discrepancy in hop {hop}, cell type {cell_type}: expected cells, but none found.")
                    return False
                elif expected_cells == 0 and actual_cells > 0:
                    # Cells found where none were expected in the TCN array
                    print(f"Discrepancy in hop {hop}, cell type {cell_type}: found cells where none were expected.")
                    return False

        # Step 2: Check that cell_type_indices agrees with the pattern for the relevant cell types
        for hop in range(num_hops):
            for cell_type_idx, cell_type in enumerate(cell_type_group):
                expected_cells = expected_pattern[hop, cell_type_idx]
                if cell_type in tcn.cell_type_indeces_dict:
                    actual_cells = len(tcn.cell_type_indeces_dict[cell_type][hop])  # Number of cells in the TCN's indices
                else:
                    actual_cells = 0  # No cells of this type in this TCN

                if expected_cells > 0 and actual_cells == 0:
                    # Expected cells but none found in cell_type_indices
                    print(f"Discrepancy in cell_type_indices for hop {hop}, cell type {cell_type}: expected cells, but none found.")
                    return False
                elif expected_cells == 0 and actual_cells > 0:
                    # Cells found in cell_type_indices where none were expected
                    print(f"Discrepancy in cell_type_indices for hop {hop}, cell type {cell_type}: found cells where none were expected.")
                    return False

        # If everything matches
        return True
    

    @staticmethod
    def remove_dependent_cells(profiles, threshold=0.9):
        """
        Drops highly correlated marker columns from `profiles`.

        Args:
            profiles: 2D array (cells × markers).
            threshold: Correlation cutoff to mark dependence.

        Returns:
            filtered_profiles: array with dependent columns removed.
            to_remove: map removed_idx→dependent_on_idx.
            independent_indices: list of kept column indices.
        """
        num_markers = profiles.shape[1]
        correlation_matrix = np.corrcoef(profiles, rowvar=False)
        to_remove = {}
        independent_indices = list(range(num_markers))

        # Identify markers with high correlation and mark them as dependent
        for i in range(num_markers):
            for j in range(i + 1, num_markers):
                if abs(correlation_matrix[i, j]) > threshold:
                    # Mark j as dependent on i if they are highly correlated
                    to_remove[j] = i
                    if j in independent_indices:
                        independent_indices.remove(j)

        # Filter out dependent markers from profiles
        filtered_profiles = profiles[:, independent_indices]
        return filtered_profiles, to_remove, independent_indices


    @staticmethod
    def analyze_markers_by_cell_type_and_hop_conservative(
        significant_combinations: Dict[Tuple[int, ...], List['TCN']], 
        idws_scores: Dict[Tuple[int, int], Dict[int, float]], 
        cell_profile_mapping: Dict[Tuple[int, int], Dict[int, np.ndarray]], 
        tcns_dict: Dict[Tuple[int, ...], List['TCN']],
        use_original_labels: bool = False, 
        idws_thresh: float = 0.5,
        csv_path: str = None, required_fracture: float = 0.5, 
        is_conservative: bool = False, 
    ) -> Dict[Tuple[int, ...], Tuple[Dict[int, Dict[str, Tuple[float, float]]], Dict[Tuple[int, int, str], Tuple[float, float]]]]:
        """
        Performs conservative per-cell-type, per-hop marker significance testing with FDR.

        Args:
            significant_combinations: Mapping from combination tuple to list of cluster TCNs.
            idws_scores: Mapping from (run_id, fov_id) to cell->IDWS score.
            cell_profile_mapping: Per-(run,fov), per-cell marker profiles.
            tcns_dict: All TCN clusters for global baseline.
            use_original_labels: If True, use TCN.label instead of IDWS.
            idws_thresh: Threshold for IDWS-based responder definitions.
            csv_path: Path to write detailed CSV; if None, skip writing.
            required_fracture: Fraction filter for marker inclusion.
            is_conservative: If True, require both responder/non-resp above threshold.

        Returns:
            Dict mapping combination to:
                - cell_type_hop_significance: hop->cell_type->(direction, -log10(p))
                - marker_p_values: (hop,marker,cell_type)->(direction,p)
        """
        
        assert 0 <= idws_thresh <= 1, "idws_thresh must be between 0 and 1."
        high_throuput_marker_mask = TCNAnalysis.filter_markers_low_throughput(significant_combinations, cell_profile_mapping)

        # IDWS thresholds
        idws_above_thresh, idws_below_thresh = TCNAnalysis.get_significant_idws_cells(idws_scores, idws_thresh)

        results = {}
        expr_results = {}

        # Calculate global (outside) distribution for all TCNs in tcns_dict
        included_cells_responder_global, included_cells_non_responder_global = TCNAnalysis.get_global_cell_ids_per_ct_hop(significant_combinations, cell_profile_mapping, use_original_labels, high_throuput_marker_mask, idws_above_thresh, idws_below_thresh)

        # Processing significant combinations
        for combination, cluster_tcns in tqdm(significant_combinations.items()):
            cell_type_hop_significance = defaultdict(lambda: defaultdict(lambda: Tuple))
            responder_markers, non_responder_markers, included_cells_responder, included_cells_non_responder = TCNAnalysis.get_cell_stat_per_ct_hop_in_comb(cell_profile_mapping, use_original_labels, high_throuput_marker_mask, idws_above_thresh, idws_below_thresh, cluster_tcns)
            marker_p_values = {}
            overall_p_values = []
            overall_indices = []
            marker_p_values_list = []
            marker_indices = []
            marker_expr_values = {}
            for hop in range(4):         
                for cell_type in responder_markers[hop].keys():
                    # Inside TCN cluster
                    p_value, direction = TCNAnalysis.get_ct_distance_p_val_direction(included_cells_responder_global, included_cells_non_responder_global, included_cells_responder, included_cells_non_responder, hop, cell_type)
                    if p_value is None:
                        continue
                    cell_type_hop_significance[hop][cell_type] = (direction, -np.log10(max(p_value, 1e-10)))
                    overall_p_values.append(p_value)
                    overall_indices.append((hop, cell_type))
                    if not non_responder_markers[hop][cell_type] or not responder_markers[hop][cell_type]:
                        continue
                    responder_profiles, non_responder_profiles, rel_marker_ind = TCNAnalysis.filter_profiles_to_rel_markers(hop, cell_type, responder_markers, non_responder_markers, high_throuput_marker_mask, required_fracture, is_conservative)
                    if responder_profiles.size > 0 and non_responder_profiles.size > 0:
                        # Store expression values for the considered markers
                        for idx, marker_idx in enumerate(rel_marker_ind):
                            marker_expr_values[(hop, marker_idx, cell_type)] = {'Responders': responder_profiles[:, idx].tolist(), 'Non-Responders': non_responder_profiles[:, idx].tolist()}
                        # Loop through each marker to assess its effect on the responder status
                        for idx, marker_idx in enumerate(rel_marker_ind):
                            # Prepare data
                            p_value, direction = TCNAnalysis.get_marker_pval_direc_mw(responder_profiles, non_responder_profiles, idx)
                            
                            # Store the result
                            marker_p_values[(hop, marker_idx, cell_type)] = (direction, p_value)
                            marker_p_values_list.append(p_value)
                            marker_indices.append((hop, marker_idx, cell_type))

                            
           # Apply FDR correction for marker tests
            if marker_p_values_list:
                _, corrected_marker_p_values, _, _ = multipletests(marker_p_values_list, method='fdr_bh')
                # Update results with FDR-corrected p-values
                for idx, corrected_p in zip(marker_indices, corrected_marker_p_values):
                    marker_p_values[idx] = (marker_p_values[idx][0], corrected_p)
            else:
                print("No marker p-values to correct.")

            # Apply FDR correction for overall cell-type tests
            if overall_p_values:
                _, corrected_ct_p_values, _, _ = multipletests(overall_p_values, method='fdr_bh')
                # Update results with FDR-corrected p-values
                for idx, corrected_p in zip(overall_indices, corrected_ct_p_values):
                    cell_type_hop_significance[idx[0]][idx[1]] = (cell_type_hop_significance[idx[0]][idx[1]][0], -np.log10(max(corrected_p, 1e-10)))
            else:
                print("No overall p-values to correct.")

            # Store results for the combination
            results[combination] = (cell_type_hop_significance, marker_p_values)
            expr_results[combination] = marker_expr_values

        return results, expr_results

    @staticmethod
    def get_marker_pval_direc_nb(responder_profiles, non_responder_profiles, idx, alpha=1e-6, maxiter=1000):
        """
        Fits a regularized Negative Binomial to test marker differential expression.

        Args:
            responder_profiles: 2D int array of responder counts per marker.
            non_responder_profiles: 2D int array for non-responders.
            idx: Index of marker column to test.
            alpha: Regularization strength for NB regression.
            maxiter: Max iterations for optimizer.

        Returns:
            Tuple (p_value, directionality) where directionality is +1 or -1.

        Raises:
            ValueError: if counts are negative or non-integer.
        """
        # Extract expression values for the specified marker
        res_profiles = responder_profiles[:, idx]
        non_res_profiles = non_responder_profiles[:, idx]

        # Combine the profiles and create group labels
        combined_profiles = np.concatenate((res_profiles, non_res_profiles))
        group_labels = np.concatenate((np.ones_like(res_profiles), np.zeros_like(non_res_profiles)))

        # Check for non-integer or negative values
        if not np.all(combined_profiles >= 0):
            raise ValueError("Expression counts must be non-negative.")
        if not np.all(np.equal(np.mod(combined_profiles, 1), 0)):
            raise ValueError("Expression counts must be integers.")

        # Remove duplicate profiles
        unique_profiles, unique_indices = np.unique(combined_profiles, return_index=True)
        unique_labels = group_labels[unique_indices]

        # Prepare the predictor matrix (X) and the response variable (y)
        X = sm.add_constant(unique_labels)  # Add intercept and group indicator
        y = unique_profiles

        # Fit a negative binomial regression model with regularization parameters
        nb_model = NegativeBinomial(y, X)
        nb_results = nb_model.fit_regularized(alpha=alpha, maxiter=maxiter, disp=False, solver="lbfgs")

        # Extract the coefficient and p-value for the group indicator
        coef = nb_results.params[1]  # Coefficient for the group indicator
        p_value = nb_results.pvalues[1]  # p-value for the group indicator

        directionality = 1 if coef > 0 else -1

        return p_value, directionality

    
    @staticmethod
    def get_marker_pval_direc_mw(responder_profiles, non_responder_profiles, idx):
        """
        Tests marker difference via Mann–Whitney U.

        Args:
            responder_profiles: 2D array of marker expressions for responders.
            non_responder_profiles: 2D array for non-responders.
            idx: Marker column index.

        Returns:
            Tuple (p_value, directionality).
        """
        # Extract expression values for the specified marker
        expr_responder = responder_profiles[:, idx]
        expr_non_responder = non_responder_profiles[:, idx]

        # Perform Mann-Whitney U test (non-parametric test)
        stat, pvalue = mannwhitneyu(expr_responder, expr_non_responder, alternative='two-sided')

        # Determine directionality based on median expression levels
        median_responder = np.median(expr_responder)
        median_non_responder = np.median(expr_non_responder)

        directionality = 1 if median_responder > median_non_responder else -1
        return pvalue, directionality


    
    @staticmethod
    def filter_profiles_to_rel_markers(hop, cell_type, responder_markers, non_responder_markers, high_throuput_marker_mask, required_fracture, is_conservative):
        """
        Filters marker profiles to those passing fraction criteria.

        Args:
            hop: Hop index.
            cell_type: Name of cell type.
            responder_markers: nested dict hop->ct->list of profiles.
            non_responder_markers: same for non-responders.
            high_throughput_marker_mask: Boolean mask of markers.
            required_fracture: fraction threshold.
            is_conservative: AND vs OR logic for threshold.

        Returns:
            Tuple (responder_profiles, non_responder_profiles, rel_marker_indices).
        """
        num_markers = len(responder_markers[0][list(responder_markers[0].keys())[0]][0])
        responder_profiles = np.array([np.asarray(p) for p in responder_markers[hop][cell_type] if len(p) == num_markers])
        non_responder_profiles = np.array([np.asarray(p) for p in non_responder_markers[hop][cell_type] if len(p) == num_markers])
        responder_expr_fraction = np.mean(responder_profiles > 0, axis=0)
        non_responder_expr_fraction = np.mean(non_responder_profiles > 0, axis=0)

        # Apply the 80% filter for each marker
        if is_conservative:
            markers_to_consider = (responder_expr_fraction > required_fracture) & (non_responder_expr_fraction > required_fracture)
        else:
            markers_to_consider = (responder_expr_fraction > required_fracture) | (non_responder_expr_fraction > required_fracture)
        rel_marker_ind = np.where(high_throuput_marker_mask)[0][markers_to_consider]
        responder_profiles = responder_profiles[:, markers_to_consider]
        non_responder_profiles = non_responder_profiles[:, markers_to_consider]
        return responder_profiles, non_responder_profiles, rel_marker_ind

    @staticmethod
    def get_ct_distance_p_val_direction(
        included_cells_responder_global, 
        included_cells_non_responder_global, 
        included_cells_responder, 
        included_cells_non_responder, 
        hop, 
        cell_type):
        """
        Fisher test and direction for a single cell-type hop contingency.

        Args:
            included_cells_responder_global: global sets per hop->ct.
            included_cells_non_responder_global: likewise for non-responders.
            included_cells_responder: inside-cluster sets.
            included_cells_non_responder: inside-cluster sets.
            hop: Hop index.
            cell_type: Cell type name.

        Returns:
            Tuple (p_value, directionality).
        """
        
        # Calculate counts of cells inside the TCN cluster
        inside_positive_count = len(included_cells_responder[hop][cell_type])
        inside_negative_count = len(included_cells_non_responder[hop][cell_type])

        # Calculate counts of cells outside the TCN cluster
        outside_positive_count = len(included_cells_responder_global[hop][cell_type]) - inside_positive_count
        outside_negative_count = len(included_cells_non_responder_global[hop][cell_type]) - inside_negative_count

        # Initialize p_value and direction to None
        p_value = None
        direction = None

        # Only proceed if there are any cells inside the TCN cluster
        if (inside_positive_count + inside_negative_count) > 0:
            contingency_table = np.array([
                [inside_positive_count, inside_negative_count],
                [outside_positive_count, outside_negative_count]
            ])
            # Perform Fisher's exact test
            _, p_value = fisher_exact(contingency_table)
            
            # Avoid division by zero in direction calculation
            if outside_positive_count > 0 and outside_negative_count > 0:
                ratio_positive = inside_positive_count / outside_positive_count
                ratio_negative = inside_negative_count / outside_negative_count
                direction = 1 if ratio_positive > ratio_negative else -1
            else:
                # Handle division by zero case
                direction = 0  # or some appropriate value or message
                print("Warning: Division by zero encountered in direction calculation.")
        else:
            print("Warning: No cells inside the TCN cluster for hop {} and cell type {}.".format(hop, cell_type))
        return p_value, direction


    @staticmethod
    def get_significant_idws_cells(idws_scores, idws_thresh):
        """
        Splits idws_scores into above-thresh and below-thresh sets.

        Args:
            idws_scores: map (run,fov)->cell->score.
            idws_thresh: absolute threshold.

        Returns:
            Tuple of two dicts for >=+thresh and <=-thresh cells.
        """
        idws_above_thresh = {
            (run_id, fov_id): {cell_id for cell_id, score in scores.items() if score >= idws_thresh}
            for (run_id, fov_id), scores in idws_scores.items()
        }
        idws_below_thresh = {
            (run_id, fov_id): {cell_id for cell_id, score in scores.items() if score <= -idws_thresh}
            for (run_id, fov_id), scores in idws_scores.items()
        }
        
        return idws_above_thresh,idws_below_thresh

    @staticmethod
    def get_global_cell_ids_per_ct_hop(significant_combinations, cell_profile_mapping, use_original_labels, high_throuput_marker_mask, idws_above_thresh, idws_below_thresh):
        """
        Aggregates global inside/outside cell ID sets for all combinations.

        Args:
            significant_combinations: map from comb->tcns.
            cell_profile_mapping: map from (run,fov)->cell->profile.
            use_original_labels: whether to use TCN.label.
            high_throughput_marker_mask: boolean mask of markers.
            idws_above_thresh/below: sets from get_significant_idws_cells.

        Returns:
            Tuple of two nested dicts: hop->ct->set of cell tuples (global inside).
        """
        included_cells_responder_global = defaultdict(lambda: defaultdict(set))
        included_cells_non_responder_global = defaultdict(lambda: defaultdict(set))
        for combination, cluster_tcns in significant_combinations.items():
            _, _, included_cells_responder_comb, included_cells_non_responder_comb = TCNAnalysis.get_cell_stat_per_ct_hop_in_comb(
                cell_profile_mapping, use_original_labels, high_throuput_marker_mask,
                idws_above_thresh, idws_below_thresh, cluster_tcns
            )
            # Update global dictionaries by merging sets within nested dictionaries
            TCNAnalysis.update_global_cell_dicts(included_cells_responder_global, included_cells_non_responder_global, included_cells_responder_comb, included_cells_non_responder_comb)
        return included_cells_responder_global,included_cells_non_responder_global

    @staticmethod
    def update_global_cell_dicts(included_cells_responder_global, included_cells_non_responder_global, included_cells_responder_comb, included_cells_non_responder_comb):
        """
        Merges combination-specific cell sets into global dictionaries.

        Args:
            included_cells_responder_global: hop->ct->set to update.
            included_cells_non_responder_global: likewise.
            included_cells_responder_comb: hop->ct->set from a cluster.
            included_cells_non_responder_comb: hop->ct->set.
        """
        for hop, cell_types in included_cells_responder_comb.items():
            for cell_type, cell_set in cell_types.items():
                included_cells_responder_global[hop][cell_type].update(cell_set)
            
        for hop, cell_types in included_cells_non_responder_comb.items():
            for cell_type, cell_set in cell_types.items():
                included_cells_non_responder_global[hop][cell_type].update(cell_set)

    @staticmethod
    def get_cell_stat_per_ct_hop_in_comb(cell_profile_mapping, use_original_labels, high_throuput_marker_mask, idws_above_thresh, idws_below_thresh, cluster_tcns):
        """
        Collects marker profiles and cell-ID sets inside a TCN cluster by hop & cell type.

        Args:
            cell_profile_mapping: map (run,fov)->cell->profile.
            use_original_labels: whether to use TCN.label.
            high_throughput_marker_mask: boolean mask.
            idws_above_thresh/below: sets of significant cells.
            cluster_tcns: list of TCNs in the cluster.

        Returns:
            Tuple of four nested dicts:
                - responder_markers: hop->ct->[profiles]
                - non_responder_markers: hop->ct->[profiles]
                - included_cells_responder: hop->ct->set of (run,fov,cell)
                - included_cells_non_responder: hop->ct->set
        """
        responder_markers = defaultdict(lambda: defaultdict(list))
        non_responder_markers = defaultdict(lambda: defaultdict(list))
        included_cells_responder = defaultdict(lambda: defaultdict(set))
        included_cells_non_responder = defaultdict(lambda: defaultdict(set))

        for tcn in cluster_tcns:
            run_id, fov_id = tcn.run_id[0], tcn.fov_id[0]
            cell_type_indices = tcn.cell_type_indeces_dict

            for hop in range(4):
                for cell_type, cell_indices in cell_type_indices.items():
                    for cell_id in cell_indices[hop]:
                        cell_profile = cell_profile_mapping[(run_id, fov_id)][cell_id]
                        if cell_profile is None:
                            continue
                        if cell_profile.ndim > 1:
                            cell_profile = cell_profile[:, high_throuput_marker_mask]
                        else:
                            cell_profile = cell_profile[high_throuput_marker_mask]
                        if use_original_labels:
                            if tcn.label[0] == 1:
                                if (run_id, fov_id, cell_id) not in included_cells_responder[hop][cell_type]:
                                    responder_markers[hop][cell_type].append(cell_profile)
                                    included_cells_responder[hop][cell_type].add((run_id, fov_id, cell_id))
                            else:
                                if (run_id, fov_id, cell_id) not in included_cells_non_responder[hop][cell_type]:
                                    non_responder_markers[hop][cell_type].append(cell_profile)
                                    included_cells_non_responder[hop][cell_type].add((run_id, fov_id, cell_id))
                        else:
                            if cell_id in idws_above_thresh.get((run_id, fov_id), set()):
                                if (run_id, fov_id, cell_id) not in included_cells_responder[hop][cell_type]:
                                    responder_markers[hop][cell_type].append(cell_profile)
                                    included_cells_responder[hop][cell_type].add((run_id, fov_id, cell_id))
                            elif cell_id in idws_below_thresh.get((run_id, fov_id), set()):
                                if (run_id, fov_id, cell_id) not in included_cells_non_responder[hop][cell_type]:
                                    non_responder_markers[hop][cell_type].append(cell_profile)
                                    included_cells_non_responder[hop][cell_type].add((run_id, fov_id, cell_id))
        return responder_markers,non_responder_markers,included_cells_responder,included_cells_non_responder

    @staticmethod
    def filter_markers_low_throughput(significant_combinations, cell_profile_mapping, precntile = 0.01):
        """
        Creates a boolean mask to drop low-throughput markers.

        Args:
            significant_combinations: map comb->tcns to index marker_tcn_arr dims.
            cell_profile_mapping: per-(run,fov) cell->profile dict.
            precntile: minimal fraction of cells expressing marker.

        Returns:
            Boolean array of length n_markers indicating keep/drop.
        """
        num_markers = significant_combinations[list(significant_combinations.keys())[0]][0].marker_tcn_arr.shape[1]
        total_marker_counts = np.zeros(num_markers)

        # Calculate overall cell expression for each marker across the dataset
        cell_count = 0 
        for profiles in cell_profile_mapping.values():
            for profile in profiles.values():
                if profile is None:
                    continue
                total_marker_counts += (profile > 0).sum(axis=0)
                cell_count += 1
        # cell_count = sum(len(cells_dict) for cells_dict in cell_profile_mapping.values())
        marker_filter = total_marker_counts >= (cell_count * precntile)
        return marker_filter



    @staticmethod
    def construct_signature(marker_results: Dict[Tuple[int, ...], Tuple[Dict[int, Dict[str, Tuple[float, float]]], Dict[Tuple[int, int, str], Tuple[float, float]]]], 
                            marker_names: List[str],  marker_expr_results: Dict,
                            top_n: int = 10,
                            csv_path: str = None) -> Dict[Tuple[int, ...], Dict[Tuple[int, str], List[Tuple[str, Tuple[float, float]]]]]:

        """Builds a top-N marker signature per combination for each cell type and hop.

        Args:
            marker_results: Mapping from combination to (cell_type_hop_significance, marker_p_values).
            marker_names: List of marker names by index.
            marker_expr_results: Expression values per combination/hop/marker.
            top_n: Number of top markers to include per cell-type/hop.
            csv_path: Optional path to save detailed signature CSV.

        Returns:
            signature: Dict mapping combination -> {(hop, cell_type): [(marker, (direction, p_value, mean_expr))...]}
        """
        
        # Create a deep copy of marker_results to prevent any unintended modifications
        marker_results_copy = copy.deepcopy(marker_results)
        expr_results_copy = copy.deepcopy(marker_expr_results)
        
        signature = defaultdict(lambda: defaultdict(list))
        signature_list = []
        combination_count = 0

        for combination, (cell_type_hop_significance, marker_p_values) in tqdm(marker_results_copy.items()):
            combination_count += 1
            for hop, cell_types in cell_type_hop_significance.items():
                for cell_type, _ in cell_types.items():
                    if cell_type_hop_significance[hop][cell_type][1] < -np.log10(0.5):
                        continue
                    sorted_markers = sorted([(marker_idx, direction, p_val) for (h, marker_idx, c_type), (direction, p_val) in marker_p_values.items() if h == hop and c_type == cell_type], key=lambda x: x[2])
                    top_markers = [(marker_names[marker_idx], (direction, p_val, np.mean(expr_results_copy[combination][(hop, marker_idx, cell_type)]['Responders' if direction == 1 else 'Non-Responders']))) for marker_idx, direction, p_val in sorted_markers[:top_n]]
                    signature[combination][(hop, cell_type)] = top_markers

                    # Collect data for CSV
                    for marker_name, (direction, p_val, expr) in top_markers:
                        signature_list.append({
                            'Combination': combination_count,
                            'Hop': hop,
                            'Cell_Type': cell_type,
                            'Marker': marker_name,
                            'Direction': 'Responders' if direction == 1 else 'Non-Responders',
                            'P_Value': p_val,
                            'Mean Expression': expr,
                        })
        
        # Convert to DataFrame and save to CSV
        if csv_path:
            signature_df = pd.DataFrame(signature_list)
            signature_df.to_csv(csv_path, index=False)
        
        return signature

    @staticmethod
    def plot_cell_type_significance(cell_type_hop_significance: Dict[int, Dict[str, Tuple[int, float]]], combination_name: str = "") -> None:
        """Saves a heatmap visualizing direction × -log10(p) per cell type and hop.

        Args:
            cell_type_hop_significance: hop->cell_type->(direction, -log10(p_value)).
            combination_name: Optional string used in output filename.

        Returns:
            None — writes ‘ct_sig_{combination_name}.svg’.
        """
        from matplotlib.colors import LinearSegmentedColormap
        # Extract hops and cell types
        hops = list(cell_type_hop_significance.keys())
        cell_types = sorted(set(cell_type for hop_data in cell_type_hop_significance.values() for cell_type in hop_data.keys()))
        
        # Create a heatmap initialized with NaN for missing data
        heatmap = np.full((len(cell_types), len(hops)), np.nan)

        for hop_idx, hop in enumerate(hops):
            for cell_type_idx, cell_type in enumerate(cell_types):
                if cell_type in cell_type_hop_significance[hop]:
                    # Multiply direction by -log10(p-value)
                    direction, minu_log_p_value = cell_type_hop_significance[hop][cell_type]
                    heatmap[cell_type_idx, hop_idx] = direction * minu_log_p_value

        # Define the data range and significance threshold
        min_value = -10
        max_value = 10
        min_significance = -np.log10(0.05)

        # Create a custom colormap with gradients on both sides
        n_colors = 256
        colors = np.ones((n_colors, 4))  # RGBA

        # Compute positions corresponding to significance thresholds
        pos_neg_sig = int((( -min_significance - min_value) / (max_value - min_value)) * n_colors)
        pos_pos_sig = int(((  min_significance - min_value) / (max_value - min_value)) * n_colors)

        # Ensure positions are within [0, n_colors - 1]
        pos_neg_sig = np.clip(pos_neg_sig, 0, n_colors - 1)
        pos_pos_sig = np.clip(pos_pos_sig, 0, n_colors - 1)

        # Gradient from blue to white (negative significant values)
        colors[0:pos_neg_sig, 0] = np.linspace(0, 1, pos_neg_sig)  # R channel from 0 to 1
        colors[0:pos_neg_sig, 1] = np.linspace(0, 1, pos_neg_sig)  # G channel from 0 to 1
        colors[0:pos_neg_sig, 2] = 1.0                             # B channel fixed at 1.0

        # White region (non-significant values)
        colors[pos_neg_sig:pos_pos_sig, 0:3] = 1.0  # RGB all set to 1.0 (white)

        # Gradient from white to red (positive significant values)
        n_red = n_colors - pos_pos_sig
        colors[pos_pos_sig:, 0] = 1.0                             # R channel fixed at 1.0
        colors[pos_pos_sig:, 1] = np.linspace(1.0, 0.0, n_red)    # G channel from 1.0 to 0.0
        colors[pos_pos_sig:, 2] = np.linspace(1.0, 0.0, n_red)    # B channel from 1.0 to 0.0

        # Create the custom colormap
        custom_cmap = LinearSegmentedColormap.from_list('CustomMap', colors)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create a new axis for the color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Display the heatmap using imshow with the custom colormap
        cax_im = ax.imshow(heatmap, cmap=custom_cmap, vmin=min_value, vmax=max_value, aspect='auto')

        # Add the color bar
        fig.colorbar(cax_im, cax=cax, label='Direction * -log10(p-value)')

        # Set ticks for hops and cell types
        ax.set_xticks(range(len(hops)))
        ax.set_yticks(range(len(cell_types)))

        ax.set_xticklabels([f'{hop}-Hop' for hop in hops], rotation=0, fontsize=14)
        ax.set_yticklabels(cell_types, fontsize=14)

        # Add minor ticks for the grid
        ax.set_xticks(np.arange(len(hops) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(cell_types) + 1) - .5, minor=True)

        # Configure grid lines for the borders
        ax.tick_params(which='minor', size=0)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"ct_sig_{combination_name}.svg", dpi=1200, facecolor='white', edgecolor='white', bbox_inches='tight')
        plt.close()



    @staticmethod
    def plot_marker_signature(signature: Dict[Tuple[int, str], List[Tuple[str, Tuple[float, float]]]], combination_name: str = "", use_expr: bool = False):
        """
        Renders a heatmap of marker significance or expression per (hop, cell_type).

        Args:
            signature: Mapping from (hop, cell_type) to [(marker, (direction, p_value or expr))].
            combination_name: Identifier for file naming.
            use_expr: If True, plot direction × mean expression instead of -log10(p).

        Returns:
            None — writes ‘marker_heatmap_{combination_name}.svg’.
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if not signature:
            print("No data to plot. The signature is empty.")
            return

        all_markers = list(set([marker for sublist in signature.values() for marker, _ in sublist]))
        if not all_markers:  # Handle case where there are no markers
            print("No markers found in the signature.")
            return
        marker_indices = {marker: idx for idx, marker in enumerate(all_markers)}

        heatmap = np.zeros((len(signature), len(all_markers)))

        for (hop, cell_type), marker_list in signature.items():
            for marker, (direction, p_value, expr) in marker_list:
                if use_expr:
                    heatmap[list(signature.keys()).index((hop, cell_type)), marker_indices[marker]] = direction * np.mean(expr)
                else:
                    heatmap[list(signature.keys()).index((hop, cell_type)), marker_indices[marker]] = direction * -np.log10(max(p_value, 1e-10))

        fig, ax = plt.subplots(figsize=(15, 20))

        # Use make_axes_locatable to create a new axis for the color bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cax_im = ax.matshow(heatmap, cmap='coolwarm', vmin=-3, vmax=3)

        # Create the color bar using the new axis
        if use_expr:
            fig.colorbar(cax_im, cax=cax, label='Direction * Mean Expression')
        else:
            fig.colorbar(cax_im, cax=cax, label='Direction * -log10(p-value)')

        ax.set_xticks(range(len(all_markers)))
        ax.set_yticks(range(len(signature.keys())))

        ax.set_xticklabels(all_markers, rotation=90, fontsize=8)
        ax.set_yticklabels([f'{cell_type} (Hop {hop})' for (hop, cell_type) in signature.keys()], fontsize=8)

        plt.xlabel('Markers')
        plt.ylabel('Cell Types and Hops')

        plt.tight_layout()

        plt.savefig(f"marker_heatmap_{combination_name}.svg", dpi=1200, facecolor='white', edgecolor='white', bbox_inches='tight')
        plt.close()
        # plt.show()

    @staticmethod
    def categorize_markers_by_pathway(
        marker_results: Dict[Tuple[int, ...], Tuple[Dict[int, Dict[str, Tuple[float, float]]], Dict[Tuple[int, int, str], Tuple[float, float]]]], 
        pathway_data: pd.DataFrame, 
        marker_names: List[str], 
        top_n_markers: int = None, 
        significance_threshold: float = 0.05, 
        combined_pvalue_threshold: float = 0.05
    ) -> Dict[Tuple[int, ...], Dict[str, Dict[Tuple[str, str, int, int, str], List[Tuple[str, str, float]]]]]:
        """
        Assigns significant markers to ligand–receptor interactions per pathway.

        Args:
            marker_results: As in construct_signature.
            pathway_data: DataFrame with columns ['Pathway','Ligand','Receptor'].
            marker_names: List of marker names.
            top_n_markers: Limit per marker, None for all.
            significance_threshold: p-value cutoff for marker inclusion.
            combined_pvalue_threshold: Fisher-combined p-value cutoff.

        Returns:
            categorized_markers: combination -> group ('Responders'/'Non-Responders') ->
                (ligand_ct,ligand_hop,receptor_ct,receptor_hop,pathway) -> [(ligand,receptor,combined_p)].
        """
        marker_results_copy = copy.deepcopy(marker_results)
        categorized_markers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for combination, (_, marker_p_values) in marker_results_copy.items():
            significant_markers = defaultdict(list)

            for (hop, marker_idx, cell_type), (direction, p_val) in marker_p_values.items():
                if p_val < significance_threshold:
                    marker_name = marker_names[marker_idx]
                    significant_markers[marker_name].append((hop, cell_type, direction, p_val))

            for pathway in pathway_data['Pathway'].unique():
                pathway_markers = pathway_data[pathway_data['Pathway'] == pathway]

                for _, row in pathway_markers.iterrows():
                    ligand = row['Ligand']
                    receptor = row['Receptor']

                    ligand_info = significant_markers.get(ligand, [])
                    receptor_info = significant_markers.get(receptor, [])

                    for ligand_hop, ligand_cell_type, ligand_direction, ligand_p_val in ligand_info:
                        for receptor_hop, receptor_cell_type, receptor_direction, receptor_p_val in receptor_info:
                            if ligand_direction == receptor_direction:
                                combined_p_val = combine_pvalues([ligand_p_val, receptor_p_val], method='fisher')[1]

                                if combined_p_val < combined_pvalue_threshold:
                                    group = 'Responders' if ligand_direction == 1 else 'Non-Responders'
                                    
                                    # Check for uniqueness in pathway recording
                                    if not categorized_markers[combination][group][(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway)]:
                                        categorized_markers[combination][group][(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway)].append((ligand, receptor, combined_p_val))

        return categorized_markers

    @staticmethod
    def get_distinct_color(existing_colors):
            """
        Selects a new color from matplotlib tab20 not in existing_colors.

        Args:
            existing_colors: Set of RGBA tuples already used.

        Returns:
            A new RGBA tuple.

        Raises:
            ValueError: if all tab20 colors are exhausted.
        """
        cmap = plt.get_cmap('tab20')  # Or use another colormap if needed
        max_colors = cmap.N  # Number of distinct colors in the colormap

        for i in range(max_colors):
            color = cmap(i / max_colors)
            if color not in existing_colors:
                return color
        
        # If all colors are used, raise an exception or extend the color palette
        raise ValueError("Ran out of distinct colors to assign!")

    @staticmethod
    def plot_network(pathways, unique_types, combination_name, type_colors=None, ax=None):
    """
        Draws a multi‐graph of cell‐type interactions on concentric hop circles.

        Args:
            pathways: mapping from pathway key -> marker list.
            unique_types: list of (ligand,receptor) pairs.
            combination_name: used for title.
            type_colors: existing ligand–receptor color map.
            ax: matplotlib Axes; if None, uses current.

        Returns:
            Updated type_colors mapping.
        """
        RADIUS = 3
        cell_type_styles = {
            "B.cell": {'shape': 's', 'color': 'red'},
            "T.CD4": {'shape': 'o', 'color': 'blue'},
            "T.CD8": {'shape': 'o', 'color': 'green'},
            "CAF": {'shape': 'd', 'color': 'purple'},
            "Endo.": {'shape': 'v', 'color': 'orange'},
            "Mal": {'shape': 'h', 'color': 'brown'},  # Mal cell
            "Macrophage": {'shape': '^', 'color': 'yellow'},
            "NK": {'shape': '>', 'color': 'pink'},
            "T.cell": {'shape': 'o', 'color': 'cyan'},
        }

        if not pathways:
            return type_colors

        if type_colors is None:
            type_colors = {}

        used_colors = set(type_colors.values())
        for ligand_receptor in unique_types:
            if ligand_receptor not in type_colors:
                color = TCNAnalysis.get_distinct_color(used_colors)
                type_colors[ligand_receptor] = color
                used_colors.add(color)

        node_shapes = {}
        G = nx.MultiDiGraph()
        node_positions = {}
        hop_distances = {0: [], 1: [], 2: [], 3: []}

        # Build the graph from pathways
        for (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway), markers in pathways.items():
            ligand_node = (ligand_cell_type, ligand_hop)
            receptor_node = (receptor_cell_type, receptor_hop)
            if ligand_node not in hop_distances[ligand_hop]:
                hop_distances[ligand_hop].append(ligand_node)
                G.add_node(ligand_node)
                node_shapes[ligand_node] = cell_type_styles.get(ligand_cell_type, {'shape': 'o', 'color': 'gray'})
            if receptor_node not in hop_distances[receptor_hop]:
                hop_distances[receptor_hop].append(receptor_node)
                G.add_node(receptor_node)
                node_shapes[receptor_node] = cell_type_styles.get(receptor_cell_type, {'shape': 'o', 'color': 'gray'})

            ligand, receptor, _ = markers[0]
            edge_color = type_colors.get((ligand, receptor), 'black')
            G.add_edge(ligand_node, receptor_node, color=edge_color)

        for hop_distance, nodes in hop_distances.items():
            total_nodes = len(nodes)
            if total_nodes > 0:
                angles = np.linspace(0, 2 * np.pi, total_nodes, endpoint=False)
                for i, node in enumerate(nodes):
                    node_pos = (hop_distance * RADIUS * np.cos(hop_distance * (0.66*np.pi ) + angles[i]), hop_distance * RADIUS * np.sin(hop_distance * (0.66*np.pi ) +angles[i]))
                    node_positions[node] = node_pos

        pos = node_positions

        # Use the provided axis `ax` for plotting
        if ax is None:
            ax = plt.gca()

        # Draw nodes
        for shape, color in set((v['shape'], v['color']) for v in node_shapes.values()):
            node_list = [node for node, style in node_shapes.items() if style['shape'] == shape and style['color'] == color]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=3000, node_color=color, node_shape=shape, ax=ax)

        # Draw edges
        for u, v, data in G.edges(data=True):
            if u == v:
                node_x, node_y = pos[u]
                loop_radius = 2.0
                arrow = FancyArrowPatch((node_x + loop_radius * 0.5, node_y + 0.3), (node_x - loop_radius * 0.5, node_y + 0.3),
                                        connectionstyle="arc3,rad=1.5", arrowstyle='-|>', color=data['color'], mutation_scale=20)
                ax.add_patch(arrow)
            else:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[data['color']], arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)

        for hop_distance in range(1, 4):
            circle = plt.Circle((0, 0), hop_distance * RADIUS, color='gray', fill=False, linestyle='--')
            ax.add_patch(circle)

        ax.set_frame_on(False)
        ax.set_title(f"Cell Type Network for {combination_name}")
        ax.axis('equal')
        return type_colors

        
    @staticmethod
    def prepare_and_plot_network(categorized_markers, combination, combination_name="", group="Responders", csv_path=None, typed_colors=None):
    """
        Produces side‐by‐side network plots for responders vs. non‐responders.

        Args:
            categorized_markers: as from categorize_markers_by_pathway.
            combination: specific combination key.
            combination_name: plot title prefix.
            group: 'Responders' or 'Non-Responders'.
            csv_path: path to append pathway CSV.
            typed_colors: initial color map.

        Returns:
            None — shows and optionally saves legend and plots.
        """
        # Filter pathways based on group
        pathways_res = categorized_markers[combination].get("Responders", {})
        pathways_nres = categorized_markers[combination].get("Non-Responders", {})

        # Identify unique ligand-receptor pairs
        unique_types_res = set()
        for (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway), markers in pathways_res.items():
            unique_types_res.update([(ligand, receptor) for ligand, receptor, _ in markers])
        unique_types_nres = set()
        for (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway), markers in pathways_nres.items():
            unique_types_nres.update([(ligand, receptor) for ligand, receptor, _ in markers])

        unique_types_res = list(unique_types_res)
        unique_types_nres = list(unique_types_nres)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Call the plotting function
        typed_colors_res = TCNAnalysis.plot_network(pathways_res, unique_types_res, combination_name + " Responders", type_colors=typed_colors, ax=ax1)
        typed_colors = TCNAnalysis.plot_network(pathways_nres, unique_types_nres, combination_name + " Non-Responders", type_colors=typed_colors_res, ax=ax2)

        # Save pathway details if csv_path is provided
        if csv_path:
            pathway_details_df_rs = pd.DataFrame(pathways_res)
            pathway_details_df_nrs = pd.DataFrame(pathways_nres)
            if os.path.exists(csv_path):
                pathway_details_df_rs.to_csv(csv_path, mode='a', header=False, index=False)
                pathway_details_df_nrs.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                pathway_details_df_rs.to_csv(csv_path, mode='w', index=False)
                pathway_details_df_nrs.to_csv(csv_path, mode='a', index=False)
        # Cell type styles (assumed defined in plot_network)
        cell_type_styles = {
            "B.cell": {'shape': 's', 'color': 'red'},
            "T.CD4": {'shape': 'o', 'color': 'blue'},
            "T.CD8": {'shape': 'o', 'color': 'green'},
            "CAF": {'shape': 'd', 'color': 'purple'},
            "Endo.": {'shape': 'v', 'color': 'orange'},
            "Mal": {'shape': 'h', 'color': 'brown'},
            "Macrophage": {'shape': '^', 'color': 'yellow'},
            "NK": {'shape': '>', 'color': 'pink'},
            "T.cell": {'shape': 'o', 'color': 'cyan'},
        }
        legend_handles = []


        # Add legend entries for cell types
        for cell_type, style in cell_type_styles.items():
            legend_handles.append(Line2D([0], [0], marker=style['shape'], color='w', label=cell_type, markerfacecolor=style['color'], markersize=10))

        # Add edge legend for ligand-receptor types
        if typed_colors is not None:
            for (ligand, receptor), color in typed_colors.items():
                legend_handles.append(Line2D([0], [0], color=color, lw=2, label=f'{ligand}-{receptor}'))

        # Add a single legend to the figure
        fig.legend(handles=legend_handles, title="Legend", loc='center right', bbox_to_anchor=(1.08, 0.5), fontsize=14)
        
        plt.tight_layout()
        # plt.savefig("pathways.svg", dpi=1200, facecolor='white', edgecolor='white', bbox_inches='tight')
        # plt.close()
        plt.show()

    @staticmethod
    def plot_global_cell_type_significance(marker_results: Dict, 
                                        significance_threshold: float = 0.05, 
                                        display_threshold: float = 0.5):
    """
        Aggregates across combinations and displays a heatmap of cell-type significance.

        Args:
            marker_results: mapping from combination to (significance & marker_pvalues).
            significance_threshold: p-level to include.
            display_threshold: p-level for display cutoff.

        Returns:
            None — shows a combined heatmap.
        """
        # Collect overall cell type significance across combinations
        marker_results_copy = copy.deepcopy(marker_results)
        cell_type_significance = defaultdict(lambda: defaultdict(list))
        cell_type_directions = defaultdict(lambda: defaultdict(list))
        
        for _, (cell_type_hop_significance, _) in tqdm(marker_results_copy.items()):
            for hop, cell_types in cell_type_hop_significance.items():
                for cell_type, (direction, log_p_value) in cell_types.items():
                    if not np.isnan(log_p_value):
                        # Convert back from -log10(p-value) to p-value
                        original_p_value = 10 ** (-log_p_value)
                        cell_type_significance[cell_type][hop].append(original_p_value)
                        cell_type_directions[cell_type][hop].append(direction)

        # Aggregate p-values using combine_pvalues_with_directions
        cell_types = sorted(cell_type_significance.keys())
        hops = sorted(set(hop for hops in cell_type_significance.values() for hop in hops))
        
        heatmap_data = np.full((len(cell_types), len(hops)), np.nan)  # Initialize heatmap with NaN

        for cell_idx, cell_type in enumerate(cell_types):
            for hop_idx, hop in enumerate(hops):
                if hop in cell_type_significance[cell_type]:
                    p_values = cell_type_significance[cell_type][hop]
                    directions = cell_type_directions[cell_type][hop]

                    # Combine p-values with directions using Stouffer's method
                    combined_z, combined_p_value = TCNAnalysis.combine_pvalues_with_directions(
                        p_values, directions, method='stouffer', weights=None, alternative='two-sided'
                    )

                    # If combined p-value is below the display threshold, set the heatmap data
                    if combined_p_value < display_threshold:
                        # Use combined_z for direction and magnitude
                        heatmap_data[cell_idx, hop_idx] = np.sign(combined_z) * -np.log10(max(combined_p_value, 1e-10))
                    # Else leave it as NaN (white)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a colormap that handles NaN values, with white representing NaN
        cmap = plt.get_cmap('coolwarm')
        cmap.set_bad(color='white')  # Set NaN to appear as white

        # Display the heatmap with a colorbar
        cax = ax.matshow(heatmap_data, cmap=cmap, vmin=-3, vmax=3)
        fig.colorbar(cax, ax=ax, label='Direction * -log10(p-value)')

        # Set tick labels for hops and cell types
        ax.set_xticks(range(len(hops)))
        ax.set_yticks(range(len(cell_types)))

        ax.set_xticklabels([f'{hop}-Hop' for hop in hops], rotation=45)
        ax.set_yticklabels(cell_types)

        # Add minor ticks for grid lines
        ax.set_xticks(np.arange(len(hops) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(cell_types) + 1) - .5, minor=True)

        # Configure grid lines for the borders
        ax.tick_params(which='minor', size=0)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

        # Final adjustments and displaying the plot
        plt.xlabel('Hops')
        plt.ylabel('Cell Types')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def combine_pvalues_with_directions(
        p_values, 
        directions, 
        method='stouffer', 
        weights=None, 
        alternative='two-sided',
        clamp_extremes=True,
        epsilon_min=1e-15,  # Minimum p-value
        epsilon_max=None,    # Maximum p-value, set to 1 - epsilon_min if None
        debug=False,         # Debug flag
    ):
        """
        Combines p-values using the specified method (currently Stouffer) with directional signs.

        Args:
            p_values: List of p-values to combine. Must match length of `directions`.
            directions: List of +1/-1 indicating effect direction per p-value.
            method: Combination method; only 'stouffer' supported.
            weights: Optional weights for each p-value; defaults to equal weights.
            alternative: 'two-sided', 'greater', or 'less'.
            clamp_extremes: If True, clamp p-values to [epsilon_min, epsilon_max].
            epsilon_min: Minimum clamp bound (default 1e-15).
            epsilon_max: Maximum clamp bound; if None, set to 1 - epsilon_min.
            debug: If True, prints debugging info during calculation.

        Returns:
            combined_z: float — combined Z-score.
            combined_p: float — two-tailed p-value corresponding to combined_z.

        Raises:
            ValueError: for invalid inputs (length mismatch, invalid p/d, non-finite values).
            NotImplementedError: if unsupported method is requested.
        """

        # Set epsilon_max if not provided
        if epsilon_max is None:
            epsilon_max = 1 - epsilon_min

        # Basic sanity checks
        if len(p_values) != len(directions):
            raise ValueError("Length of p_values and directions must be the same.")

        if method.lower() != 'stouffer':
            raise NotImplementedError(f"Method '{method}' is not implemented. Only 'stouffer' is supported.")

        z_scores = []
        for idx, (p, d) in enumerate(zip(p_values, directions)):
            # 1) Check for None or NaN
            if p is None or d is None:
                raise ValueError(f"p_values or directions contained a None entry at index {idx}.")

            if np.isnan(p) or np.isnan(d):
                raise ValueError(f"p_values or directions contained a NaN entry at index {idx}.")

            # 2) Clamp p-values to [epsilon_min, epsilon_max] if clamp_extremes is True
            original_p = p  # For debugging
            if clamp_extremes:
                if p < epsilon_min:
                    p = epsilon_min
                    if debug:
                        print(f"Clamped p-value at index {idx} from {original_p} to {p}")
                elif p > epsilon_max:
                    p = epsilon_max
                    if debug:
                        print(f"Clamped p-value at index {idx} from {original_p} to {p}")
            else:
                # Ensure p is within (0,1) to avoid infinities
                if not (0 < p < 1):
                    raise ValueError(f"Invalid p-value: {p} at index {idx}. Must be strictly between 0 and 1.")

            # 3) Convert to z-score based on alternative
            try:
                if alternative == 'two-sided':
                    # Two-tailed
                    z = norm.ppf(1 - p / 2)
                    # Ensure direction is strictly +1 or -1
                    sign_d = np.sign(d)
                    if sign_d not in [-1, 1]:
                        raise ValueError(f"Invalid direction {d} at index {idx}. Must be +1 or -1.")
                    z *= sign_d

                elif alternative == 'greater':
                    # One-tailed: p-value is for the positive direction
                    if d < 0:
                        # direction < 0 => p is for negative side
                        z = norm.ppf(p)
                    else:
                        z = norm.ppf(1 - p)

                elif alternative == 'less':
                    # One-tailed: p-value is for the negative direction
                    if d > 0:
                        # direction > 0 => p is for positive side
                        z = norm.ppf(p)
                    else:
                        z = norm.ppf(1 - p)

                else:
                    raise ValueError("Invalid value for 'alternative'. Choose from 'two-sided', 'greater', 'less'.")

                if not np.isfinite(z):
                    raise ValueError(f"Non-finite Z-score computed at index {idx} with p={p} and d={d}.")

            except Exception as e:
                raise ValueError(f"Error computing Z-score for p-value {p} and direction {d} at index {idx}: {e}")

            z_scores.append(z)

        z_scores = np.array(z_scores, dtype=float)

        if debug:
            print(f"Computed Z-scores: {z_scores}")

        # Apply weights
        if weights is not None:
            if len(weights) != len(z_scores):
                raise ValueError("Length of weights must match length of p_values.")
            weights = np.array(weights, dtype=float)
            # Check for None or NaN in weights
            if np.any(np.isnan(weights)):
                raise ValueError("Weights contain NaN.")
            if np.any(weights == None):
                raise ValueError("Weights contain None.")
        else:
            weights = np.ones_like(z_scores)

        if debug:
            print(f"Using weights: {weights}")

        # Check that weights are finite
        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights contain non-finite values (inf or NaN).")

        # Check that not all weights are zero
        if not np.any(weights):
            raise ValueError("All weights are zero, cannot compute a valid combined Z-score.")

        # Check that all z_scores are finite
        if not np.all(np.isfinite(z_scores)):
            raise ValueError("Some Z-scores are non-finite (inf or NaN). Cannot combine.")

        # Compute the weighted sum of Z-scores
        weighted_z_sum = np.sum(weights * z_scores)

        if debug:
            print(f"Weighted sum of Z-scores: {weighted_z_sum}")

        # Compute sqrt of sum of weights^2
        sum_weights_sq = np.sqrt(np.sum(weights**2))

        if debug:
            print(f"Sum of weights squared (sqrt): {sum_weights_sq}")

        if sum_weights_sq == 0:
            raise ValueError("Sum of squares of weights is zero, cannot compute combined Z.")

        # Combined Z-score
        combined_z = weighted_z_sum / sum_weights_sq

        if not np.isfinite(combined_z):
            raise ValueError(f"Combined Z-score is non-finite: {combined_z}")

        # Two-tailed p-value for the combined Z
        combined_p = 2 * norm.sf(abs(combined_z))

        if not np.isfinite(combined_p):
            raise ValueError(f"Combined p-value is non-finite: {combined_p}")

        if debug:
            print(f"Combined Z: {combined_z}, Combined p: {combined_p}")

        return combined_z, combined_p




    @staticmethod
    def plot_global_pathway_network(categorized_markers, typed_colors=None, cell_types_to_plot=None, restrict_mal_to_center=False, plot_unique_pathways_only=False):
        """
        Plots side-by-side global ligand–receptor networks for responders vs. non-responders.

        Args:
            categorized_markers: Output of `categorize_markers_by_pathway`.
            typed_colors: Optional existing color map for ligand/receptor pairs.
            cell_types_to_plot: If provided, restrict nodes to these cell types.
            restrict_mal_to_center: If True, only include 'Mal' at hop 0.
            plot_unique_pathways_only: If True, exclude pathways common to both groups.

        Returns:
            None — displays a Matplotlib figure with two network subplots.
        """
        all_pathways_responders = {}
        all_pathways_non_responders = {}
        unique_types_responders = set()
        unique_types_non_responders = set()

        # Function to check if a pathway interaction involves the specified cell types
        def is_relevant_interaction(ligand_cell, receptor_cell):
            if cell_types_to_plot is None:
                return True
            return ligand_cell in cell_types_to_plot and receptor_cell in cell_types_to_plot

        # Function to check if "Mal" cells obey the restriction
        def is_mal_restricted(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop):
            if not restrict_mal_to_center:
                return True
            if ligand_cell_type == "Mal" and ligand_hop != 0:
                return False
            if receptor_cell_type == "Mal" and receptor_hop != 0:
                return False
            return True

        # Loop through all combinations and collect pathways for each group
        for combination in categorized_markers:
            # Collect all pathways for 'Responders'
            responders_pathways = categorized_markers[combination].get('Responders', {})
            for (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway), markers in responders_pathways.items():
                if is_relevant_interaction(ligand_cell_type, receptor_cell_type) and is_mal_restricted(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop):
                    if (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway) not in all_pathways_responders:
                        all_pathways_responders[(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway)] = markers
                    unique_types_responders.update([(ligand, receptor) for ligand, receptor, _ in markers])

            # Collect all pathways for 'Non-Responders'
            non_responders_pathways = categorized_markers[combination].get('Non-Responders', {})
            for (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway), markers in non_responders_pathways.items():
                if is_relevant_interaction(ligand_cell_type, receptor_cell_type) and is_mal_restricted(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop):
                    if (ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway) not in all_pathways_non_responders:
                        all_pathways_non_responders[(ligand_cell_type, ligand_hop, receptor_cell_type, receptor_hop, pathway)] = markers
                    unique_types_non_responders.update([(ligand, receptor) for ligand, receptor, _ in markers])

            # Filter unique pathways if flag is enabled
        if plot_unique_pathways_only:
            # flip_path = lambda expr: (expr[0], expr[1], expr[2], expr[3], "->".join(expr[4].split("->")[::-1]))
            # common_pathways = set(all_pathways_responders.keys()).intersection(all_pathways_non_responders.keys())
            # all_pathways_responders = {key: val for key, val in all_pathways_responders.items() if key not in common_pathways or flip_path(key) not in common_pathways}
            # all_pathways_non_responders = {key: val for key, val in all_pathways_non_responders.items() if key not in common_pathways or flip_path(key) not in common_pathways}
            flip_pathway = lambda expr: "->".join(expr.split("->")[::-1])
            common_pathways = set()
            responders_keys = set(all_pathways_responders.keys())
            non_responders_keys = set(all_pathways_non_responders.keys())

            # Check for both direct and flipped matches
            for pathway in responders_keys:
                key = pathway[4]
                flipped_pathway = flip_pathway(key)
                for nr_pathway in non_responders_keys:
                    if nr_pathway[4] == key or nr_pathway[4] == flipped_pathway:
                        common_pathways.add(key)
                        break

            # Remove common pathways from both groups
            all_pathways_responders = {key: val for key, val in all_pathways_responders.items() if key[4] not in common_pathways}
            all_pathways_non_responders = {key: val for key, val in all_pathways_non_responders.items() if key[4] not in common_pathways}
            unique_types_non_responders = [val for val in unique_types_non_responders if f"{val[0]}->{val[1]}" not in common_pathways and f"{val[1]}->{val[0]}" not in common_pathways]
            unique_types_responders = [val for val in unique_types_responders if f"{val[0]}->{val[1]}" not in common_pathways and f"{val[1]}->{val[0]}" not in common_pathways]
            unique_types_non_responders = set(unique_types_non_responders)
            unique_types_responders = set(unique_types_responders)

        # Convert unique types sets to lists
        unique_types_responders = list(unique_types_responders)
        unique_types_non_responders = list(unique_types_non_responders)
        print(f"Unique types responders: {unique_types_responders}")
        print(f"Unique types non-responders: {unique_types_non_responders}")

        # Create a side-by-side layout for responders and non-responders
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot for Responders
        print("Plotting Global Network for Responders")
        typed_colors_res = TCNAnalysis.plot_network(all_pathways_responders, unique_types_responders, "Responders", type_colors=typed_colors, ax=ax1)

        # Plot for Non-Responders
        print("Plotting Global Network for Non-Responders")
        typed_colors = TCNAnalysis.plot_network(all_pathways_non_responders, unique_types_non_responders, "Non-Responders", type_colors=typed_colors_res, ax=ax2)

        # Create legend handles for cell types and ligand-receptor pairs
        legend_handles = []

        # Cell type styles (assumed defined in plot_network)
        cell_type_styles = {
            "B.cell": {'shape': 's', 'color': 'red'},
            "T.CD4": {'shape': 'o', 'color': 'blue'},
            "T.CD8": {'shape': 'o', 'color': 'green'},
            "CAF": {'shape': 'd', 'color': 'purple'},
            "Endo.": {'shape': 'v', 'color': 'orange'},
            "Mal": {'shape': 'h', 'color': 'brown'},
            "Macrophage": {'shape': '^', 'color': 'yellow'},
            "NK": {'shape': '>', 'color': 'pink'},
            "T.cell": {'shape': 'o', 'color': 'cyan'},
        }

        if cell_types_to_plot is not None:
            cell_type_styles = { k:v for k, v in cell_type_styles.items() if k in cell_types_to_plot}
        # Add legend entries for cell types
        for cell_type, style in cell_type_styles.items():
            legend_handles.append(Line2D([0], [0], marker=style['shape'], color='w', label=cell_type, markerfacecolor=style['color'], markersize=10))

        # Add edge legend for ligand-receptor types
        if typed_colors is not None:
            for (ligand, receptor), color in typed_colors.items():
                legend_handles.append(Line2D([0], [0], color=color, lw=2, label=f'{ligand}-{receptor}'))

        # Add a single legend to the figure
        fig.legend(handles=legend_handles, title="Legend", loc='center right', bbox_to_anchor=(1.08, 0.5), fontsize=14)
        
        plt.tight_layout()
        plt.show()


    @staticmethod
    def get_comb_markers_dict_with_details(
        marker_names: List[str], 
        significance_threshold: float, 
        marker_results_copy: Dict[str, Tuple[bool, Dict[Tuple[int, int, str], Tuple[int, float]]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Aggregates combined marker z-scores and p-values with detailed metadata.

        Args:
            marker_names: List of all marker names.
            significance_threshold: Threshold for p-values to include.
            marker_results_copy: Deep copy of marker results mapping.

        Returns:
            combined_marker_results: Dict mapping marker name ->
                {'z': combined_z, 'p': combined_p, 'z_scores': [...], 'details': [...]}
        """
        marker_significance = defaultdict(lambda: {'p_values': [], 'directions': [], 'details': []})
        combined_marker_results = defaultdict(dict)
        
        # Define clamping parameters
        epsilon_min = 1e-15
        epsilon_max = 1 - epsilon_min
        combination_number = 0
        for combination, (_, marker_p_values) in marker_results_copy.items():
            combination_number += 1
            for (hop, marker_idx, cell_type), (direction, p_value) in marker_p_values.items():
                # Validate marker_idx
                if marker_idx >= len(marker_names):
                    print(f"Warning: marker_idx {marker_idx} out of range for marker_names.")
                    continue

                marker_name = marker_names[marker_idx]

                # Validate p_value
                if p_value is None:
                    print(f"Warning: p_value is None for marker '{marker_name}' in combination '{combination}'. Skipping.")
                    continue
                try:
                    p_value = float(p_value)
                    if np.isnan(p_value):
                        print(f"Warning: p_value is NaN for marker '{marker_name}' in combination '{combination}'. Skipping.")
                        continue
                    if p_value < epsilon_min:
                        p_value = epsilon_min  # Prevent p_value=0
                        # print(f"Clamped p_value for marker '{marker_name}' in combination '{combination}' from 0 to {epsilon_min}")
                    elif p_value > epsilon_max:
                        p_value = epsilon_max  # Prevent p_value=1
                        # print(f"Clamped p_value for marker '{marker_name}' in combination '{combination}' from >1 to {epsilon_max}")
                except (ValueError, TypeError):
                    raise ValueError(f"Warning: Invalid p_value '{p_value}' for marker '{marker_name}' in combination '{combination}'. Skipping.")

                # Validate direction
                if direction not in [1, -1]:
                    print(f"Warning: Invalid direction '{direction}' for marker '{marker_name}' in combination '{combination}'. Skipping.")
                    continue

                marker_significance[marker_name]['p_values'].append(p_value)
                marker_significance[marker_name]['directions'].append(direction)
                marker_significance[marker_name]['details'].append((combination_number, cell_type, hop))

        # Combine p-values with directions for each marker
        for marker, data in marker_significance.items():
            p_values = data['p_values']
            directions = data['directions']
            details = data['details']

            # Ensure there are p-values and directions to combine
            if not p_values or not directions:
                print(f"Info: No valid p-values or directions for marker '{marker}'. Skipping combination.")
                continue

            try:
                # Convert to z-scores with clamping
                z_scores = []
                for idx, (p, d) in enumerate(zip(p_values, directions)):
                    # Compute z-score based on two-sided test
                    z = norm.ppf(1 - p / 2)
                    z *= d  # Apply direction

                    if not np.isfinite(z):
                        print(f"Excluding non-finite z-score for marker '{marker}': z={z}, detail={details[idx]}")
                        continue

                    z_scores.append(z)

                if not z_scores:
                    print(f"No finite z-scores for marker '{marker}'. Skipping combination.")
                    continue

                # Combine z-scores using Stouffer's method
                combined_z, combined_p = TCNAnalysis.combine_pvalues_with_directions(
                    p_values, directions, method='stouffer', alternative='two-sided', clamp_extremes=True, epsilon_min=epsilon_min, epsilon_max=epsilon_max, debug=False
                )

                # Ensure combined_z and combined_p are valid
                if combined_z is None or combined_p is None:
                    print(f"Info: Combined z or p is None for marker '{marker}'. Skipping.")
                    continue
                if not (np.isfinite(combined_z) and np.isfinite(combined_p)):
                    print(f"Info: Combined z or p is not finite for marker '{marker}'. Skipping.")
                    continue

                # Store combined results with details
                combined_marker_results[marker] = {
                    'z': combined_z,
                    'p': combined_p,
                    'z_scores': z_scores,
                    'details': details  # You might want to store only finite z_scores' details
                }

            except Exception as e:
                print(f"Error combining p-values for marker '{marker}': {e}")
                continue

        return combined_marker_results
    
    @staticmethod
    def plot_top_marker_significance_with_direction_bar(
        marker_results: Dict[str, Tuple[bool, Dict[Tuple[int, int, str], Tuple[int, float]]]], 
        marker_names: List[str], 
        top_n_responders: int = 10, 
        top_n_nonresponders: int = 10, 
        significance_threshold: float = 0.05, 
    ):
        """
        Plots horizontal bar charts of top N markers by combined z-score for responders/non-responders.

        Args:
            marker_results: Mapping from combination to (flag, marker_p_values).
            marker_names: List of marker names.
            top_n_responders: Number of top responder markers to plot.
            top_n_nonresponders: Number of top non-responder markers.
            significance_threshold: p-value cutoff for inclusion.

        Returns:
            None — writes 'top_markers_significance_with_direction_bar_plot.svg'.
        """
        
        # Deep copy to prevent modifications to original data
        marker_results_copy = copy.deepcopy(marker_results)
        
        # Collect p-values and directions across all combinations with combination details
        combined_marker_results = TCNAnalysis.get_comb_markers_dict_with_details(
            marker_names, significance_threshold, marker_results_copy
        )
        
        if not combined_marker_results:
            print("No combined marker results to plot.")
            return
        
        # Filter out markers with invalid combined z-scores or p-values
        valid_marker_results = {}
        for marker, res in combined_marker_results.items():
            if np.isfinite(res['z']) and np.isfinite(res['p']):
                valid_marker_results[marker] = res
            else:
                print(f"Excluding marker '{marker}' due to invalid combined z-score or p-value.")
        
        if not valid_marker_results:
            print("No valid combined marker results to plot after filtering.")
            return
        
        # Separate markers into Responders (positive z) and Non-Responders (negative z)
        responders_markers = {marker: res for marker, res in valid_marker_results.items() if res['z'] > 0}
        non_responders_markers = {marker: res for marker, res in valid_marker_results.items() if res['z'] < 0}
        
        # Sort markers within each category based on combined z-score absolute value
        sorted_responders = sorted(
            responders_markers.items(), 
            key=lambda x: abs(x[1]['z']), 
            reverse=True
        )[:top_n_responders]
        
        sorted_non_responders = sorted(
            non_responders_markers.items(), 
            key=lambda x: abs(x[1]['z']), 
            reverse=True
        )[:top_n_nonresponders]
        
        print("Top Responders:", sorted_responders)
        print("Top Non-Responders:", sorted_non_responders)
        
        # Extract marker names and z-scores
        top_markers_responders = [marker for marker, _ in sorted_responders]
        top_z_responders = [res['z'] for _, res in sorted_responders]
        
        top_markers_non_responders = [marker for marker, _ in sorted_non_responders]
        top_z_non_responders = [res['z'] for _, res in sorted_non_responders]
        
        # Create a single figure with two columns: Responders and Non-Responders
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(15, max(top_n_responders, top_n_nonresponders) * 0.5 + 3),
            constrained_layout=True
        )
        
        # Plot Responders
        ax_responder = axes[0]
        if top_markers_responders:
            y_positions = np.arange(len(top_markers_responders))
            bars = ax_responder.barh(y_positions, top_z_responders, color='lightcoral', edgecolor='black')
            ax_responder.set_yticks(y_positions)
            ax_responder.set_yticklabels(top_markers_responders, fontsize=10)
            ax_responder.invert_yaxis()  # Highest value at top
            ax_responder.set_xlabel("Combined Z-Score", fontsize=12)
            ax_responder.set_title(f"Top {top_n_responders} Responders", fontsize=14)
            ax_responder.grid(True, axis='x', linestyle='--', alpha=0.7)
            # ax_responder.set_xlim(0,top_z_responders[0]*1.2)
            
            # Annotate bars with z-scores
            for bar, z in zip(bars, top_z_responders):
                width = bar.get_width()
                ax_responder.annotate(f"{z:.2f}",
                                      xy=(width, bar.get_y() + bar.get_height() / 2),
                                      xytext=(-5, 0),  # Offset text by 5 points to the right
                                      textcoords="offset points",
                                      va='center',
                                      ha='right',
                                      fontsize=10)
        else:
            ax_responder.text(0.5, 0.5, 'No Responders', ha='center', va='center', fontsize=12)
            ax_responder.set_title("Responders", fontsize=14)
            ax_responder.set_xticks([])
            ax_responder.set_yticks([])
        
        # Plot Non-Responders
        ax_non_responder = axes[1]
        if top_markers_non_responders:
            y_positions = np.arange(len(top_markers_non_responders))
            bars = ax_non_responder.barh(y_positions, top_z_non_responders, color='skyblue', edgecolor='black')
            ax_non_responder.set_yticks(y_positions)
            ax_non_responder.set_yticklabels(top_markers_non_responders, fontsize=10)
            ax_non_responder.invert_yaxis()  # Highest value at top
            ax_non_responder.set_xlabel("Combined Z-Score", fontsize=12)
            ax_non_responder.set_title(f"Top {top_n_nonresponders} Non-Responders", fontsize=14)
            ax_non_responder.grid(True, axis='x', linestyle='--', alpha=0.7)
            # ax_non_responder.set_xlim(top_z_non_responders[0]*1.2, 0)  
            ax_non_responder.invert_xaxis()

            # Annotate bars with z-scores
            for bar, z in zip(bars, top_z_non_responders):
                width = bar.get_width()
                ax_non_responder.annotate(f"{z:.2f}",
                                          xy=(width, bar.get_y() + bar.get_height() / 2),
                                          xytext=(-5, 0),  # Offset text by 5 points to the left
                                          textcoords="offset points",
                                          va='center',
                                          ha='right',
                                          fontsize=10)
        else:
            ax_non_responder.text(0.5, 0.5, 'No Non-Responders', ha='center', va='center', fontsize=12)
            ax_non_responder.set_title("Non-Responders", fontsize=14)
            ax_non_responder.set_xticks([])
            ax_non_responder.set_yticks([])
        
        # # Save the figure
        plt.savefig("top_markers_significance_with_direction_bar_plot.svg", dpi=1200, 
                    facecolor='white', edgecolor='white', bbox_inches='tight')
        plt.close(fig)
        # plt.show()
        
    @staticmethod
    def plot_top_marker_significance_with_direction(
        marker_results: Dict[str, Tuple[bool, Dict[Tuple[int, int, str], Tuple[int, float]]]], 
        marker_names: List[str], 
        marker_expr_results: Dict = None,  # Deprecated, kept for compatibility
        top_n_responders: int = 10, 
        top_n_nonresponders: int = 10, 
        significance_threshold: float = 0.05, 
        include_scatter: bool = True, 
        sort_by_difference: bool = False, 
        use_expr: bool = False  # Deprecated, kept for compatibility
    ):
        """
        Creates side-by-side violin-and-scatter plots of top markers for responders vs. non-responders.

        Args:
            marker_results: Mapping as above.
            marker_names: List of names indexed by marker index.
            marker_expr_results: Deprecated.
            top_n_responders: Count for responders.
            top_n_nonresponders: Count for non-responders.
            significance_threshold: p-value cutoff.
            include_scatter: If True, overlay individual points.
            sort_by_difference: If True, sort by difference in z-scores.
            use_expr: Deprecated; if True uses expression instead of -log10(p).

        Returns:
            None — writes 'top_markers_combined_violin_enhanced_matplotlib.svg'.
        """
        
        # Deep copy to prevent modifications to original data
        marker_results_copy = copy.deepcopy(marker_results)
        marker_expr_results_copy = copy.deepcopy(marker_expr_results) if marker_expr_results else {}
        
        # Collect p-values and directions across all combinations with combination details
        combined_marker_results = TCNAnalysis.get_comb_markers_dict_with_details(
            marker_names, significance_threshold, marker_results_copy
        )
        
        if not combined_marker_results:
            print("No combined marker results to plot.")
            return
        
        # Filter out markers with invalid combined z-scores or p-values
        valid_marker_results = {}
        for marker, res in combined_marker_results.items():
            if np.isfinite(res['z']) and np.isfinite(res['p']):
                valid_marker_results[marker] = res
            else:
                print(f"Excluding marker '{marker}' due to invalid combined z-score or p-value.")
        
        if not valid_marker_results:
            print("No valid combined marker results to plot after filtering.")
            return
        
        # Separate markers into positive and negative based on combined z-score
        positive_markers = {marker: res for marker, res in valid_marker_results.items() if res['z'] > 0}
        negative_markers = {marker: res for marker, res in valid_marker_results.items() if res['z'] < 0}
        
        # Sort markers within each category based on combined z-score absolute value
        sorted_positive_markers = sorted(
            positive_markers.items(), 
            key=lambda x: abs(x[1]['z']), 
            reverse=True
        )[:top_n_responders]
        
        sorted_negative_markers = sorted(
            negative_markers.items(), 
            key=lambda x: abs(x[1]['z']), 
            reverse=True
        )[:top_n_nonresponders]
        
        print("Top Positive Markers:", sorted_positive_markers)
        print("Top Negative Markers:", sorted_negative_markers)
        
        # Extract marker names
        top_markers_positive = [marker for marker, _ in sorted_positive_markers]
        top_markers_negative = [marker for marker, _ in sorted_negative_markers]
        
        # Function to plot on a given axis using Violin Plots
        def plot_violin_with_scatter(
            ax, 
            marker_list: List[str], 
            combined_z_scores: Dict[str, float], 
            marker_z_scores: Dict[str, Dict[str, List[float]]], 
            direction: str, 
            title: str,
            include_scatter:bool = True,
            max_extremes:int = 4
        ):
            """
            Plot violin plots with optional scatter points.

            Parameters:
            - ax (matplotlib.axes.Axes): The axes to plot on.
            - marker_list (List[str]): List of markers to plot.
            - combined_z_scores (Dict[str, float]): Combined z-scores for markers.
            - marker_z_scores (Dict[str, Dict[str, List[float]]]): Individual z-scores for markers.
            - direction (str): 'Responders' or 'Non Responders'.
            - title (str): Title for the subplot.
            """
            ax2 = ax.twinx()
            # Prepare data: list of lists containing absolute z-scores
            data = []
            valid_marker_list = []
            for marker in marker_list:
                z_scores = marker_z_scores[marker]['z_scores']
                # Filter out non-finite z-scores (already handled, but double-check)
                finite_z_scores = [z for z in z_scores if np.isfinite(z)]
                if finite_z_scores:
                    data.append(finite_z_scores)
                    valid_marker_list.append(marker)
                else:
                    print(f"Skipping marker '{marker}' as it has no finite z-scores.")
            
            if not data:
                print(f"No valid data to plot for {direction}.")
                return
            
            # Create violin plots
            parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
            
            # Customize violin plots
            color = 'lightcoral' if direction == 'Responders' else 'skyblue'
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            for i, marker in enumerate(valid_marker_list):
                z_scores = [z for z in marker_z_scores[marker]['z_scores'] if np.isfinite(z)]
                ax.scatter(
                    np.full(len(z_scores), i + 1),
                    z_scores,
                    color='black',
                    alpha=0.6,
                    s=10
                )
            
            # Plot combined z-scores as bold horizontal lines (absolute value)
            for i, marker in enumerate(valid_marker_list):
                combined_z = combined_z_scores.get(marker, 0)
                ax2.hlines(combined_z, i + 0.75, i + 1.25, colors='k', linestyles='-', linewidth=2)
            
            # Initialize a dictionary to track y positions for each x position
            annotations_y = defaultdict(list)

            # Define minimum distance between annotations to prevent overlap
            # Adjust this value based on the scale of your z-scores
            min_dist = 0.2
            offset_step = min_dist

            # Iterate over each marker to add annotations
            for i, marker in enumerate(valid_marker_list):
                z_scores = marker_z_scores[marker]['z_scores']
                details = marker_z_scores[marker].get('details', [])
                
                # Sort indices based on the absolute value of z-scores in descending order
                sorted_indices = sorted(range(len(z_scores)), key=lambda idx: abs(z_scores[idx]), reverse=True)
                
                # Select the top 'max_extremes' indices with the highest absolute z-scores
                extreme_indices = sorted_indices[:max_extremes]
                
                for idx in extreme_indices:
                    # Construct the combination detail string
                    if idx < len(details):
                        combination_detail = f"TCN_{details[idx][0]}_ct_{details[idx][1]}_hop_{details[idx][2]}"
                    else:
                        combination_detail = "TCN_Unknown"
                    
                    # Determine the vertical alignment based on direction and z-score sign
                    if direction == 'Responders':
                        alignment = 'top' if z_scores[idx] > 0 else 'bottom'
                    else:
                        alignment = 'top' if z_scores[idx] < 0 else 'bottom'
                    
                    x_pos = i + 1  # x-position for the annotation (assuming violin plots are 1-indexed)
                    y_pos = z_scores[idx]  # Original y-position based on z-score
                    
                    # Initialize adjusted_y with the original y_pos
                    adjusted_y = y_pos
                    num_offsets = 0  # Counter for how many offsets have been applied
                    
                    # Adjust y_pos to prevent overlap
                    while any(abs(existing_y - adjusted_y) < min_dist for existing_y in annotations_y[x_pos]):
                        num_offsets += 1
                        if alignment == 'top':
                            # Shift upwards if alignment is 'top'
                            adjusted_y = y_pos + offset_step * num_offsets
                        else:
                            # Shift downwards if alignment is 'bottom'
                            adjusted_y = y_pos - offset_step * num_offsets
                    
                    # Record the adjusted y-position to keep track for future annotations
                    annotations_y[x_pos].append(adjusted_y)
                    
                    # Add the annotation with the adjusted y-position
                    ax.text(
                        x_pos,
                        adjusted_y,
                        combination_detail,  # Label with combination detail
                        horizontalalignment='center',
                        verticalalignment=alignment,
                        fontsize=8,
                        color='black',
                        rotation=45
                    )

            
            ax.set_xticks(range(1, len(valid_marker_list) + 1))
            ax.set_xticklabels(valid_marker_list, rotation=0, ha='right', fontsize=10)
            ax.set_ylabel('Individual Z-Score', fontsize=16)
            ax2.set_ylabel('Combined Z-Score', fontsize=16, color='k')
            ax2.tick_params(axis='y', colors='k')
            # ax.set_title(title, fontsize=16)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            if direction == 'Non Responders':
                ax.invert_yaxis()
                ax2.invert_yaxis()
            individual_z_proxy = Line2D([0], [0], marker='o', color='w', label='Individual Z-Score',
                                        markerfacecolor='black', markersize=8, markeredgecolor='w')

            # Combined Z-Scores (Lines)
            combined_z_proxy = Line2D([0], [0], color='k', lw=2, label='Combined Z-Score')

            # Add the legend to the bottom left of the plot
            ax.legend(handles=[individual_z_proxy, combined_z_proxy], loc='lower left', fontsize=12)

            plt.tight_layout()

        # Create side-by-side plots for responders and non-responders with constrained_layout
        fig, axes = plt.subplots(1, 2, figsize=(20, 12), constrained_layout=True)
        
        # Plot Responders
        if top_markers_positive:
            plot_violin_with_scatter(
                ax=axes[0],
                marker_list=top_markers_positive,
                combined_z_scores={marker: res['z'] for marker, res in positive_markers.items()},
                marker_z_scores=positive_markers,
                direction='Responders',
                title=f'Top {top_n_responders} Responders',
                include_scatter = include_scatter
            )
        else:
            axes[0].text(0.5, 0.5, 'No Responders', ha='center', va='center')
            axes[0].set_title('Responders', fontsize=16)
            axes[0].set_xticks([])
        
        # Plot Non-Responders
        if top_markers_negative:
            plot_violin_with_scatter(
                ax=axes[1],
                marker_list=top_markers_negative,
                combined_z_scores={marker: res['z'] for marker, res in negative_markers.items()},
                marker_z_scores=negative_markers,
                direction='Non Responders',
                title=f'Top {top_n_nonresponders} Non-Responders',
                include_scatter = include_scatter
            )
        else:
            axes[1].text(0.5, 0.5, 'No Non-Responders', ha='center', va='center')
            axes[1].set_title('Non Responders', fontsize=16)
            axes[1].set_xticks([])
        
        plt.savefig("top_markers_combined_violin_enhanced_matplotlib.svg", dpi=1200, facecolor='white', edgecolor='white', bbox_inches='tight')
        plt.close()  


    @staticmethod
    def get_all_marker_expressions_by_cell_type_and_label(
        tcns_dict: Dict[Tuple[int, ...], List['TCN']],
        cell_profile_mapping: Dict[Tuple[int, int], Dict[int, np.ndarray]],
        marker_names: List[str],
        idws_scores: Dict[Tuple[int, int], Dict[int, float]],
        num_markers: int = 960,
        idws_thresh: float = 0.5
    ) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
            """
        Gathers per-cell expression lists for both original labels and IDWS classes per cell-type.

        Args:
            tcns_dict: cluster mapping.
            cell_profile_mapping: per-(run,fov) cell->profile.
            marker_names: list of marker names.
            idws_scores: per-cell IDWS scores.
            num_markers: expected marker count per profile.
            idws_thresh: threshold for defining responder/non-responder.

        Returns:
            - all_marker_expressions_orig: cell_type -> (run,fov) -> marker->values
            - all_marker_expressions_idws: same for IDWS-defined groups
            - key_to_label_dict: map (run,fov)-> 'Responders'/'Non Responders'
        """
        
        all_marker_expressions_orig = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        all_marker_experssion_idws = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        idws_above_thresh, idws_below_thresh = TCNAnalysis.get_significant_idws_cells(idws_scores, idws_thresh)
        key_to_label_dict = {}
        for combination, cluster_tcns in tcns_dict.items():
            for tcn in cluster_tcns:
                run_id, fov_id = tcn.run_id[0], tcn.fov_id[0]
                cell_type_indices = tcn.cell_type_indeces_dict
                label = 'Responders' if tcn.label[0] == 1 else 'Non Responders'
                if (run_id, fov_id) not in key_to_label_dict:
                    key_to_label_dict[(run_id, fov_id)] = label
                included_cells_responder = defaultdict(lambda: defaultdict(set))
                included_cells_non_responder = defaultdict(lambda: defaultdict(set))

                for hop in range(4):  # Iterate over all hops
                    for cell_type, cell_indices in cell_type_indices.items():
                        for cell_id in cell_indices[hop]:
                            # Fetch the cell profile for the marker expressions
                            cell_profile = cell_profile_mapping.get((run_id, fov_id), {}).get(cell_id)
                            if cell_profile is None:
                                continue
                            cell_profile = cell_profile.flatten()
                            if len(cell_profile) != num_markers:
                                print(f"Length mismatch: expected {num_markers}, got {len(cell_profile)}")
                                continue
                                                        # Append each marker's expression to the relevant dictionary under the appropriate label
                            for marker_idx, expression_value in enumerate(cell_profile):
                                marker_name = marker_names[marker_idx]
                                all_marker_expressions_orig[cell_type][(run_id, fov_id)][marker_name].append(expression_value)
                            if cell_id in idws_above_thresh.get((run_id, fov_id), set()) and label == 'Responders':
                                if (run_id, fov_id, cell_id) not in included_cells_responder[hop][cell_type]:
                                    for marker_idx, expression_value in enumerate(cell_profile):
                                        marker_name = marker_names[marker_idx]
                                        all_marker_experssion_idws[cell_type][(run_id, fov_id)][marker_name].append(expression_value)
                            elif cell_id in idws_below_thresh.get((run_id, fov_id), set()) and label == 'Non Responders':
                                if (run_id, fov_id, cell_id) not in included_cells_non_responder[hop][cell_type]:
                                    for marker_idx, expression_value in enumerate(cell_profile):
                                        marker_name = marker_names[marker_idx]
                                        all_marker_experssion_idws[cell_type][(run_id, fov_id)][marker_name].append(expression_value)                   
        return all_marker_expressions_orig, all_marker_experssion_idws, key_to_label_dict




    def calculate_individual_z_scores(marker_results, cell_type, marker_names, epsilon_min = 1e-15):
        """
        Computes per-marker, per-hop individual z-scores across combinations for one cell type.

        Args:
            marker_results: mapping from combination to significance data.
            cell_type: cell-type to filter.
            marker_names: list of marker names.
            epsilon_min: minimum p-value for clamping.

        Returns:
            - marker_z_scores: marker_name -> hop -> list of z-scores
            - marker_details: marker_name -> hop -> list of (combination_label, z_score)
        """
        marker_z_scores = defaultdict(lambda: defaultdict(list))
        marker_details = defaultdict(lambda: defaultdict(list))
        combination_counter = 1  # Initialize combination counter
        epsilon_max = 1 - epsilon_min
        for combination, (cell_type_hop_significance, marker_p_values) in marker_results.items():
            combination_counter += 1
            for (hop, marker_idx, ct), (direction, p_value) in marker_p_values.items():
                if ct != cell_type:
                    continue  # Skip other cell types

                marker_name = marker_names[marker_idx]

                # Calculate individual z-score
                try:
                    if p_value < epsilon_min:
                        p_value = epsilon_min
                    elif p_value > epsilon_max:
                        p_value = epsilon_max

                    # Compute z-score based on two-sided test
                    z = norm.ppf(1 - p_value / 2)
                    z_score = direction * z
                    if np.isnan(z_score) or not np.isfinite(z_score):
                        raise ValueError
                except Exception as e:
                    print(f"Error computing z-score for marker {marker_name}, hop {hop}, combination {combination}: {e}")
                    continue

                marker_z_scores[marker_name][hop].append(z_score)

                # Assign combination number
                tcn_label = f"TCN_{combination_counter}"
                marker_details[marker_name][hop].append((tcn_label, z_score))

        return marker_z_scores, marker_details

    @staticmethod
    def sort_markers(marker_z_scores: Dict[str, Dict[int, List[float]]], top_n: int, cell_type_to_plot: str) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[str, float], Dict[str, float]]:
        """
        Sorts markers based on combined z-scores per hop and selects top N for responders and non-responders.

        Parameters:
        - marker_z_scores (Dict[str, Dict[int, List[float]]]): 
            Dictionary where keys are marker names and values are dictionaries mapping hop to list of z-scores.
        - top_n (int): Number of top markers to select for responders and non-responders.
        - cell_type_to_plot (str): Cell type to determine relevant hops.

        Returns:
        - top_markers_responder (Dict[int, List[str]]): Dictionary with hop as keys and list of top markers for responders as values.
        - top_markers_non_responder (Dict[int, List[str]]): Dictionary with hop as keys and list of top markers for non-responders as values.
        - combined_z_scores_responder (Dict[str, float]): Dictionary with marker names as keys and combined z-scores as values for responders.
        - combined_z_scores_non_responder (Dict[str, float]): Dictionary with marker names as keys and combined z-scores as values for non-responders.
        """
        top_markers_responder = defaultdict(list)
        top_markers_non_responder = defaultdict(list)
        combined_z_scores_responder = defaultdict(float)
        combined_z_scores_non_responder = defaultdict(float)

        # Determine relevant hops based on cell_type_to_plot
        if cell_type_to_plot == "Mal":
            relevant_hops = [0]
        else:
            relevant_hops = [3, 2, 1]

        for hop in relevant_hops:
            # Collect combined z-scores per marker for the current hop
            combined_z_scores = {}

            for marker, hops_dict in marker_z_scores.items():
                if hop not in hops_dict:
                    continue
                z_scores = hops_dict[hop]

                if not z_scores:
                    continue  # No z-scores to combine

                try:
                    combined_z = TCNAnalysis.combine_z_scores(z_scores, method='stouffer')
                except Exception as e:
                    raise ValueError(f"Error combining z-scores for marker '{marker}' at hop {hop}: {e}")

                combined_z_scores[marker] = combined_z

            # Separate markers into responders and non-responders based on combined z-score
            responders = {marker: z for marker, z in combined_z_scores.items() if z > 0}
            non_responders = {marker: z for marker, z in combined_z_scores.items() if z < 0}

            # Sort responders by absolute combined z-score descending
            sorted_responders = sorted(responders.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            top_markers_responder[hop] = [marker for marker, z in sorted_responders]
            combined_z_scores_responder.update({marker: z for marker, z in sorted_responders})

            # Sort non-responders by absolute combined z-score descending
            sorted_non_responders = sorted(non_responders.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            top_markers_non_responder[hop] = [marker for marker, z in sorted_non_responders]
            combined_z_scores_non_responder.update({marker: z for marker, z in sorted_non_responders})

        return top_markers_responder, top_markers_non_responder, combined_z_scores_responder, combined_z_scores_non_responder

    @staticmethod
    def combine_z_scores(z_scores: List[float], method: str = 'stouffer', weights: List[float] = None) -> float:
        """
        Combines a list of z-scores into one via Stouffer's method.

        Args:
            z_scores: list of z-scores.
            method: only 'stouffer' supported.
            weights: optional weights; defaults to equal.

        Returns:
            combined_z: float result.

        Raises:
            ValueError: for non-finite inputs or weight mismatches.
        """
        if method.lower() == 'stouffer':
            if weights is None:
                weights = np.ones(len(z_scores))
            else:
                weights = np.array(weights, dtype=float)
                if len(weights) != len(z_scores):
                    raise ValueError("Length of weights must match length of z_scores.")

            z_scores = np.array(z_scores, dtype=float)
            weights = np.array(weights, dtype=float)

            # Check for non-finite values
            if not np.all(np.isfinite(z_scores)):
                raise ValueError("All z-scores must be finite.")
            if not np.all(np.isfinite(weights)):
                raise ValueError("All weights must be finite.")

            weighted_sum = np.sum(weights * z_scores)
            sum_weights_sq = np.sqrt(np.sum(weights ** 2))

            if sum_weights_sq == 0:
                raise ValueError("Sum of squares of weights is zero, cannot compute combined Z.")

            combined_z = weighted_sum / sum_weights_sq
            return combined_z
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented. Only 'stouffer' is supported.")


    @staticmethod
    def plot_top_marker_sign_cell_type(marker_results, marker_names,
                                       cell_type_to_plot,
                                       top_n=10,
                                       significance_threshold=0.05,
                                       sort_by_difference=False):
        """
        Plots the top N markers for a given cell type using violin plots of z-scores with overlaid scatter plots.
        Markers are separated by hop, and top markers per hop are selected.

        Args:
            marker_results: mapping from combination to marker data.
            marker_names: list of marker names.
            cell_type_to_plot: which cell type to restrict.
            top_n: markers per hop.
            significance_threshold: p-value cutoff.
            sort_by_difference: unused.
            
        Returns:
            (fig_responder, fig_non_responder): Matplotlib figures.
        """

        @staticmethod
        def plot_violin_with_scatter(
            ax: plt.Axes,
            marker_list: List[str],
            combined_z_scores: Dict[str, float],
            marker_z_scores: Dict[str, Dict[int, List[float]]],
            marker_details: Dict[str, Dict[int, List[str]]],
            hop: int,
            direction: str,
            color: str,
            combined_color: str = 'magenta',
            max_extremes: int = 4
        ):
            """
            Plots violin plots with overlaid scatter points for a list of markers on a given axis.
            Additionally, plots combined z-scores on a separate y-axis with distinct colors and annotations.

            Parameters:
            - ax (matplotlib.axes.Axes): The primary axis to plot individual z-scores on.
            - marker_list (List[str]): List of marker names to plot.
            - combined_z_scores (Dict[str, float]): Combined z-score for each marker.
            - marker_z_scores (Dict[str, Dict[int, List[float]]]): Dictionary of marker z-scores per hop.
            - marker_details (Dict[str, Dict[int, List[str]]]): Dictionary of marker details for outlier labeling.
            - hop (int): The current hop being plotted.
            - direction (str): 'Responders' or 'Non Responders'.
            - color (str): Color for the individual z-scores (violin plots and scatter points).
            - combined_color (str): Color for the combined z-scores (lines and annotations).
            - max_extremes (int): Number of extreme annotations per marker.

            Returns:
            - None
            """
            
            # Create a secondary y-axis for combined z-scores
            ax2 = ax.twinx()

            # Prepare data for violin plots
            data = []
            valid_marker_list = []
            for marker in marker_list:
                if marker not in marker_z_scores or hop not in marker_z_scores[marker]:
                    print(f"No z-scores found for marker '{marker}' at hop {hop}. Skipping.")
                    continue
                z_scores = marker_z_scores[marker][hop]
                if not z_scores:
                    print(f"No z-scores for marker '{marker}' at hop {hop}. Skipping.")
                    continue

                # Depending on direction, filter z-scores
                if direction == 'Responders':
                    z_plot = [z for z in z_scores if z > 0]
                else:
                    z_plot = [z for z in z_scores if z < 0]

                if not z_plot:
                    print(f"No {direction.lower()} z-scores for marker '{marker}' at hop {hop}. Skipping.")
                    continue

                data.append(z_plot)
                valid_marker_list.append(marker)

            if not data:
                print(f"No valid data to plot for {direction} at hop {hop}.")
                return

            # Create violin plots on primary y-axis
            parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

            # Customize violin plots
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            # Overlay scatter points on primary y-axis
            for i, marker in enumerate(valid_marker_list):
                z_scores = marker_z_scores[marker][hop]
                if direction == 'Responders':
                    z_values = [z for z in z_scores if z > 0]
                else:
                    z_values = [z for z in z_scores if z < 0]

                for z in z_values:
                    ax.scatter(
                        i + 1,
                        z,
                        color='black',
                        alpha=0.6,
                        s=30,
                        edgecolor='w',
                        linewidth=0.5
                    )

            # Plot combined z-scores on secondary y-axis as bold horizontal lines
            for i, marker in enumerate(valid_marker_list):
                combined_z = combined_z_scores.get(marker, 0)
                ax2.hlines(combined_z, i + 0.75, i + 1.25, colors=combined_color, linestyles='-', linewidth=2)

            # Set labels and title
            ax.set_xticks(range(1, len(valid_marker_list) + 1))
            ax.set_xticklabels(valid_marker_list, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Individual Z-Score', fontsize=16, color='black')
            ax.set_title(f'Hop {hop}', fontsize=16)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            # Customize secondary y-axis
            ax2.set_ylabel('Combined Z-Score', fontsize=16, color=combined_color)
            ax2.tick_params(axis='y', colors=combined_color)

            # Invert y-axis for Non Responders on both axes
            if direction == 'Non Responders':
                ax.invert_yaxis()
                ax2.invert_yaxis()

            # ------------------ Adding the Legend ------------------

            # Create proxy artists for the legend
            # Individual Z-Scores (Dots)
            individual_z_proxy = Line2D([0], [0], marker='o', color='w', label='Individual Z-Score',
                                        markerfacecolor='black', markersize=8, markeredgecolor='w')

            # Combined Z-Scores (Lines)
            combined_z_proxy = Line2D([0], [0], color=combined_color, lw=2, label='Combined Z-Score')

            # Add the legend to the bottom left of the plot
            ax.legend(handles=[individual_z_proxy, combined_z_proxy], loc='lower left', fontsize=12)

            # ------------------ End of Legend Addition ------------------

            # Adjust layout to prevent overlap
            plt.tight_layout()
            

        # ---------------------- Main Function Logic ----------------------

        # Calculate individual z-scores and marker details
        marker_z_scores, marker_details = TCNAnalysis.calculate_individual_z_scores(marker_results, cell_type_to_plot, marker_names)

        if not marker_z_scores:
            print(f"No markers found for cell type {cell_type_to_plot}.")
            return None, None

        # Sort markers and get top markers per hop for responders and non-responders
        top_markers_responder, top_markers_non_responder, combined_z_responder, combined_z_non_responder = TCNAnalysis.sort_markers(marker_z_scores, top_n, cell_type_to_plot)

        # Determine relevant hops
        if cell_type_to_plot == "Mal":
            relevant_hops = [0]
        else:
            relevant_hops = [3, 2, 1]

        # Plot Responders Figure
        fig_responder, axes_responder = plt.subplots(1, len(relevant_hops), figsize=(5 * len(relevant_hops), 6), sharey=True)
        if len(relevant_hops) == 1:
            axes_responder = [axes_responder]  # Ensure iterable

        for idx, hop in enumerate(relevant_hops):
            markers = top_markers_responder.get(hop, [])
            if not markers:
                axes_responder[idx].text(0.5, 0.5, 'No markers', ha='center', va='center')
                axes_responder[idx].set_title(f'Hop {hop}')
                axes_responder[idx].set_xticks([])
                continue

            plot_violin_with_scatter(
                ax=axes_responder[idx],
                marker_list=markers,
                combined_z_scores=combined_z_responder,
                marker_z_scores=marker_z_scores,
                marker_details=marker_details,
                hop=hop,
                direction='Responders',
                color='lightcoral'
            )

        # fig_responder.suptitle(f'Top {top_n} Responders for {cell_type_to_plot}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Plot Non-Responders Figure
        fig_non_responder, axes_non_responder = plt.subplots(1, len(relevant_hops), figsize=(5 * len(relevant_hops), 6), sharey=True)
        if len(relevant_hops) == 1:
            axes_non_responder = [axes_non_responder]  # Ensure iterable

        for idx, hop in enumerate(relevant_hops):
            markers = top_markers_non_responder.get(hop, [])
            if not markers:
                axes_non_responder[idx].text(0.5, 0.5, 'No markers', ha='center', va='center')
                axes_non_responder[idx].set_title(f'Hop {hop}')
                axes_non_responder[idx].set_xticks([])
                continue

            plot_violin_with_scatter(
                ax=axes_non_responder[idx],
                marker_list=markers,
                combined_z_scores=combined_z_non_responder,
                marker_z_scores=marker_z_scores,
                marker_details=marker_details,
                hop=hop,
                direction='Non Responders',
                color='skyblue'
            )

        # fig_non_responder.suptitle(f'Top {top_n} Non-Responders for {cell_type_to_plot}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig_responder, fig_non_responder


    @staticmethod
    def plot_top_marker_sign_cell_type_bar(marker_results: Dict,
                                        marker_names: List[str],
                                        cell_type_to_plot: str,
                                        top_n: int = 10,
                                        fig_width: int = 3):
        """
        Plots the top N markers for a given cell type using horizontal bar plots of 
        sorted combined z-scores, separated by hop for responders and non-responders.
        """
        # Step 1: Calculate individual z-scores
        marker_z_scores, marker_details = TCNAnalysis.calculate_individual_z_scores(
            marker_results, cell_type_to_plot, marker_names
        )

        if not marker_z_scores:
            print(f"No markers found for cell type {cell_type_to_plot}.")
            return None

        # Step 2: Sort markers and get top markers per hop for responders and non-responders
        (top_markers_responder,
        top_markers_non_responder,
        combined_z_responder,
        combined_z_non_responder) = TCNAnalysis.sort_markers(
            marker_z_scores, top_n, cell_type_to_plot
        )

        # Step 3: Determine relevant hops
        if cell_type_to_plot == "Mal":
            relevant_hops = [0]
        else:
            relevant_hops = [3, 2, 1]

        # Create figure with two columns: responders (left) and non-responders (right)
        fig, axes = plt.subplots(
            nrows=len(relevant_hops),
            ncols=2,
            figsize=(fig_width * 2, 2.5 * len(relevant_hops)),
            constrained_layout=True
        )

        # Ensure axes is a 2D array
        if len(relevant_hops) == 1:
            axes = np.array([axes])

        for idx, hop in enumerate(relevant_hops):
            # -- RESPONDERS PLOT --
            ax_responder = axes[idx][0]
            markers_responder = top_markers_responder.get(hop, [])
            ax_responder.set_title(f'Hop {hop}', fontsize=12)

            if not markers_responder:
                ax_responder.text(0.5, 0.5, 'No markers', ha='center', va='center')
                ax_responder.set_yticks([])
            else:
                # Retrieve combined z-scores and sort them
                combined_z_responder_hop = {marker: combined_z_responder[marker] for marker in markers_responder}
                sorted_markers = sorted(combined_z_responder_hop.items(), key=lambda x: x[1], reverse=True)
                markers_sorted = [m for m, _ in sorted_markers]
                z_sorted = [z for _, z in sorted_markers]

                # Plot horizontal bars
                y_positions = np.arange(len(markers_sorted))
                bars = ax_responder.barh(y_positions, z_sorted, color='lightcoral', edgecolor='black')

                # Label the y-axis with marker names
                ax_responder.set_yticks(y_positions)
                ax_responder.set_yticklabels(markers_sorted, fontsize=10)
                ax_responder.invert_yaxis()  # Highest value at top
                ax_responder.grid(True, axis='x', linestyle='--', alpha=0.7)

                # Show numerical values next to each bar
                for bar, val in zip(bars, z_sorted):
                    width = bar.get_width()
                    x_offset = -5
                    ha = 'right'
                    ax_responder.annotate(f"{val:.2f}",
                                        xy=(width, bar.get_y() + bar.get_height() / 2),
                                        xytext=(x_offset, 0),
                                        textcoords="offset points",
                                        va='center', ha=ha, fontsize=9)

                ax_responder.set_xlabel("Combined Z-Score", fontsize=10)

            # -- NON-RESPONDERS PLOT --
            ax_non_responder = axes[idx][1]
            markers_non_responder = top_markers_non_responder.get(hop, [])
            ax_non_responder.set_title(f'Hop {hop}', fontsize=12)

            if not markers_non_responder:
                ax_non_responder.text(0.5, 0.5, 'No markers', ha='center', va='center')
                ax_non_responder.set_yticks([])
            else:
                # Retrieve combined z-scores and sort them ascending
                combined_z_non_responder_hop = {marker: combined_z_non_responder[marker] for marker in markers_non_responder}
                sorted_markers = sorted(combined_z_non_responder_hop.items(), key=lambda x: x[1])  # Ascending
                markers_sorted = [m for m, _ in sorted_markers]
                z_sorted = [z for _, z in sorted_markers]
                # Plot horizontal bars
                y_positions = np.arange(len(markers_sorted))
                bars = ax_non_responder.barh(y_positions, z_sorted, color='skyblue', edgecolor='black')

                ax_non_responder.set_yticks(y_positions)
                ax_non_responder.set_yticklabels(markers_sorted, fontsize=10)
                ax_non_responder.invert_xaxis()  # Invert x-axis
                ax_non_responder.invert_yaxis()  # Invert y-axis
                ax_non_responder.grid(True, axis='x', linestyle='--', alpha=0.7)

                # Annotate each bar with its value
                for bar, val in zip(bars, z_sorted):
                    width = bar.get_width()
                    x_offset = -5
                    ha = 'right'
                    ax_non_responder.annotate(f"{val:.2f}",
                                            xy=(width, bar.get_y() + bar.get_height() / 2),
                                            xytext=(x_offset, 0),
                                            textcoords="offset points",
                                            va='center', ha=ha, fontsize=9)

                ax_non_responder.set_xlabel("Combined Z-Score", fontsize=10)

        return fig


    @staticmethod
    def get_group_expr_values(marker_names, marker_expr_results_copy):
        """
        Gathers full expression lists by cell type and hop for box/violin plotting.

        Args:
            marker_names: list of names.
            marker_expr_results_copy: map combination to expr dict.

        Returns:
            Tuple of two nested dicts: responder and non-responder expr lists.
        """
        # Retain the full list of values for box plot distribution
        marker_expr_responder = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        marker_expr_non_responder = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))              
        for combination, marker_expr in marker_expr_results_copy.items():
            for (hop, marker_idx, cell_type), expr_dict in marker_expr.items():
                if hop == 0 and cell_type != 'Mal':  # Skip hop 0
                    continue
                if cell_type == 'Mal' and hop != 0:  
                    continue                
                marker_name = marker_names[marker_idx]

                # Append each value to maintain the full distribution
                marker_expr_responder[cell_type][hop][marker_name].extend(expr_dict['Responders'])
                marker_expr_non_responder[cell_type][hop][marker_name].extend(expr_dict['Non-Responders'])
        return marker_expr_responder, marker_expr_non_responder
 
    @staticmethod
    def get_marker_box_plots(direction, color, include_scatter, ax, responder_expr, non_responder_expr, use_violin_plot=True):
        """
        Creates violin or box plots with optional scatter overlays for marker exprs.

        Args:
            direction: 'Responders' or 'Non Responders'.
            color: color name for group.
            include_scatter: whether to show data points.
            ax: Matplotlib Axes to draw on.
            responder_expr: list of arrays per marker.
            non_responder_expr: similarly.
            use_violin_plot: if False, uses boxplots.

        Returns:
            None — draws on provided ax.
        """
        if use_violin_plot:
            # Violin plot for responders and non-responders
            if direction == 'Responders':
                vp1 = ax.violinplot(responder_expr, positions=np.arange(len(responder_expr)) - 0.2, widths=0.3, showmeans=False, showmedians=True, showextrema=True)
                vp2 = ax.violinplot(non_responder_expr, positions=np.arange(len(non_responder_expr)) + 0.2, widths=0.3, showmeans=False, showmedians=True, showextrema=True)
                # Set colors for the violins
                for body in vp1['bodies']:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.7)
                for body in vp2['bodies']:
                    body.set_facecolor('darkgray')
                    body.set_edgecolor('darkgray')
                    body.set_alpha(0.9)
                for key in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians']:
                    try:
                        vp1[key].set_color('dark'+ color)
                        vp1[key].set_edgecolor('dark'+ color)
                        vp2[key].set_color('darkblue')
                        vp2[key].set_edgecolor('darkblue')
                        vp1[key].set_alpha(0.7)
                        vp2[key].set_alpha(0.7)
                    except KeyError:
                        pass

            else:
                vp1 = ax.violinplot(responder_expr, positions=np.arange(len(responder_expr)) - 0.2, widths=0.3, showmeans=False, showmedians=True, showextrema=True)
                vp2 = ax.violinplot(non_responder_expr, positions=np.arange(len(non_responder_expr)) + 0.2, widths=0.3, showmeans=False, showmedians=True, showextrema=True)
                
                # Set colors for the violins
                for body in vp1['bodies']:
                    body.set_facecolor('darkgray')
                    body.set_edgecolor('darkgray')
                    body.set_alpha(0.9)
                for body in vp2['bodies']:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.7)
                for key in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians']:
                    try:
                        vp2[key].set_color('dark'+ color)
                        vp2[key].set_edgecolor('dark'+ color)
                        vp1[key].set_color('darkred')
                        vp1[key].set_edgecolor('darkred')
                        vp1[key].set_alpha(0.7)
                        vp2[key].set_alpha(0.7)
                    except KeyError:
                        pass

        else:
            # Boxplot logic as provided
            if direction == 'Responders':
                bp1 = ax.boxplot(responder_expr, positions=np.arange(len(responder_expr)) - 0.2, widths=0.3,
                                 patch_artist=True, boxprops=dict(facecolor=color, color=color),
                                 medianprops=dict(color='dark' + color), showfliers=include_scatter, notch=True, manage_ticks=False)
                
                bp2 = ax.boxplot(non_responder_expr, positions=np.arange(len(non_responder_expr)) + 0.2, widths=0.3,
                                 patch_artist=True, boxprops=dict(facecolor='gray', color='gray'),
                                 medianprops=dict(color='darkgray'), showfliers=include_scatter, notch=True, manage_ticks=False)
            else:
                bp1 = ax.boxplot(responder_expr, positions=np.arange(len(responder_expr)) - 0.2, widths=0.3,
                                 patch_artist=True, boxprops=dict(facecolor='gray', color='gray'),
                                 medianprops=dict(color='darkgray'), showfliers=include_scatter, notch=True, manage_ticks=False)
                
                bp2 = ax.boxplot(non_responder_expr, positions=np.arange(len(non_responder_expr)) + 0.2, widths=0.3,
                                 patch_artist=True, boxprops=dict(facecolor=color, color=color),
                                 medianprops=dict(color='dark' + color), showfliers=include_scatter, notch=True, manage_ticks=False)
                

    @staticmethod
    def build_dataset(responder_markers, non_responder_markers, top_markers):
        """
        Builds feature matrix X and label vector y using selected markers.

        Args:
            responder_markers: map marker->list of expr for responders.
            non_responder_markers: similarly for non-responders.
            top_markers: markers to include in dataset.

        Returns:
            X: 2D array of shape (n_samples, n_markers).
            y: 1D array of 0/1 labels.
        """
        # Collect unique lengths
        responder_lengths = []
        non_responder_lengths = []

        for marker in top_markers:
            if marker in responder_markers:
                responder_lengths.append(len(responder_markers[marker]))
            if marker in non_responder_markers:
                non_responder_lengths.append(len(non_responder_markers[marker]))

        # Debug: Print unique lengths
        print(f"Unique lengths in responder markers: {set(responder_lengths)}")
        print(f"Unique lengths in non-responder markers: {set(non_responder_lengths)}")

        # Original dataset-building logic follows here
        X_responder = []
        X_non_responder = []

        for marker in top_markers:
            if marker in responder_markers:
                data = responder_markers[marker]
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    X_responder.append(np.array(data))
                else:
                    X_responder.append(np.zeros(len(responder_markers[next(iter(responder_markers))])))
            else:
                X_responder.append(np.zeros(len(responder_markers[next(iter(responder_markers))])))

        for marker in top_markers:
            if marker in non_responder_markers:
                data = non_responder_markers[marker]
                if isinstance(data, list) or isinstance(data, np.ndarray):
                    X_non_responder.append(np.array(data))
                else:
                    X_non_responder.append(np.zeros(len(non_responder_markers[next(iter(non_responder_markers))])))
            else:
                X_non_responder.append(np.zeros(len(non_responder_markers[next(iter(non_responder_markers))])))

        X_responder = np.column_stack(X_responder)
        X_non_responder = np.column_stack(X_non_responder)

        X = np.vstack([X_responder, X_non_responder])
        y = np.array([1] * X_responder.shape[0] + [0] * X_non_responder.shape[0])

        return X, y



    @staticmethod
    def train_and_evaluate_classifier_with_kfold(X, y, n_splits=5):
        """
        Trains and evaluates a logistic-regression model via KFold CV.

        Args:
            X: feature matrix.
            y: target labels.
            n_splits: number of folds.

        Returns:
            mean_auc: mean ROC-AUC across folds.
            std_auc: standard deviation of ROC-AUC.
        """
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, classification_report

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            # Split the data into training and testing sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train logistic regression model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Predict probabilities for ROC-AUC
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            auc_scores.append(auc)

            # Optional: print metrics for each fold
            y_pred = model.predict(X_test)
            print(f"Fold {fold_idx + 1} Classification Report:")
            print(classification_report(y_test, y_pred))
            print(f"Fold {fold_idx + 1} AUC-ROC: {auc:.3f}")
            print("-" * 50)

        # Aggregate results
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)

        print(f"Mean AUC-ROC across {n_splits} folds: {mean_auc:.3f}")
        print(f"Standard Deviation of AUC-ROC: {std_auc:.3f}")

        return mean_auc, std_auc
    
    @staticmethod
    def classify_ct_with_kfold(marker_names, significance_threshold, marker_results, marker_expr_results, number_of_markers, n_splits=5):
        """
        Runs end-to-end top-N marker selection and classifier evaluation.

        Args:
            marker_names: all marker names.
            significance_threshold: p-value cutoff.
            marker_results: mapping for p-values/directions.
            marker_expr_results: expression data.
            number_of_markers: top-N features to use.
            n_splits: CV folds.

        Returns:
            mean_auc, std_auc as in train_and_evaluate_classifier.
        """
        # Deep copy of the results to avoid modifying the original data
        marker_results_copy = copy.deepcopy(marker_results)
        marker_expr_results_copy = copy.deepcopy(marker_expr_results)

        # Get combined marker significance and expressions
        marker_expression_responder, marker_expression_non_responder, combined_markers = TCNAnalysis.get_comb_markers_dict_and_expr(
            marker_names, significance_threshold, marker_results_copy, marker_expr_results_copy
        )

        # Select top markers
        top_markers_dict = sorted(combined_markers.items(), key=lambda x: abs(x[1]['z']), reverse=True)[:number_of_markers]
        top_markers = [marker for marker, _ in top_markers_dict]

        # Build dataset
        X, y = TCNAnalysis.build_dataset(marker_expression_responder, marker_expression_non_responder, top_markers)

        # Train and evaluate classifier using K-Fold Cross-Validation
        mean_auc, std_auc = TCNAnalysis.train_and_evaluate_classifier_with_kfold(X, y, n_splits=n_splits)

        return mean_auc, std_auc
    
    @staticmethod
    def get_comb_markers_dict_and_expr_per_marker(marker_names, significance_threshold, marker_results_copy, marker_expr_results_copy):
        """
        Combines p-values and aggregates expression for each marker across combinations.

        Args:
            marker_names: list of names.
            significance_threshold: p cutoff.
            marker_results_copy: mapping for p-values.
            marker_expr_results_copy: mapping for expr data.

        Returns:
            Tuple:
                responder_expr: cell_type->marker->list
                non_responder_expr: same
                combined_marker_results: cell_type->marker->{'z','p'}
        """
        marker_significance = defaultdict(lambda: defaultdict(lambda: {'p_values': [], 'directions': []}))
        marker_expression_responder = defaultdict(lambda: defaultdict(list))
        marker_expression_non_responder = defaultdict(lambda: defaultdict(list))
        
        for combination, (_, marker_p_values) in marker_results_copy.items():
            for (hop, marker_idx, cell_type), (direction, p_value) in marker_p_values.items():
                # Ensure p_value is not None and is a valid float
                if p_value is not None:
                    try:
                        p_value_float = float(p_value)
                        if np.isnan(p_value_float):
                            print(f"p_value is NaN for marker index {marker_idx} in cell type '{cell_type}'. Skipping.")
                            continue
                    except (ValueError, TypeError):
                        print(f"Invalid p_value '{p_value}' for marker index {marker_idx} in cell type '{cell_type}'. Skipping.")
                        continue
                else:
                    print(f"p_value is None for marker index {marker_idx} in cell type '{cell_type}'. Skipping.")
                    continue

                # Ensure direction is valid (assuming direction should be either 1 or -1)
                if direction not in [1, -1]:
                    print(f"Invalid direction '{direction}' for marker index {marker_idx} in cell type '{cell_type}'. Skipping.")
                    continue
                marker_name = marker_names[marker_idx]
                marker_significance[cell_type][marker_name]['p_values'].append(p_value_float)
                marker_significance[cell_type][marker_name]['directions'].append(direction)
                
                expr_data = marker_expr_results_copy.get(combination, {}).get((hop, marker_idx, cell_type), {})
                if direction == 1:
                    responders = expr_data.get('Responders', [])
                    if responders is not None:
                        marker_expression_responder[cell_type][marker_name].extend(responders)
                    else:
                        print(f"Responders data is None for marker '{marker_name}' in cell type '{cell_type}'.")
                else:
                    non_responders = expr_data.get('Non-Responders', [])
                    if non_responders is not None:
                        marker_expression_non_responder[cell_type][marker_name].extend(non_responders)
                    else:
                        print(f"Non-Responders data is None for marker '{marker_name}' in cell type '{cell_type}'.")

        # Combine p-values with directions for each marker
        combined_marker_results = defaultdict(dict)
        for cell_type, marker_dict in marker_significance.items():
            for marker, data in marker_dict.items():
                p_values = data['p_values']
                directions = data['directions']
                
                # Ensure there are p-values and directions to combine
                if not p_values or not directions:
                    print(f"No valid p-values or directions for marker '{marker}' in cell type '{cell_type}'. Skipping combination.")
                    continue

                # Optional: Check for minimum number of p-values required for combination
                if len(p_values) < 2:
                    print(f"Not enough p-values for marker '{marker}' in cell type '{cell_type}' to perform combination. Skipping.")
                    continue

                try:
                    combined_z, combined_p = TCNAnalysis.combine_pvalues_with_directions(
                        p_values, directions, method='stouffer', alternative='two-sided'
                    )
                    
                    # Ensure combined_z and combined_p are not None
                    if combined_z is None or combined_p is None:
                        print(f"Combined z or p is None for marker '{marker}' in cell type '{cell_type}'. Skipping.")
                        continue

                    # Check if combined_z and combined_p are finite
                    if not (np.isfinite(combined_z) and np.isfinite(combined_p)):
                        print(f"Combined z or p is not finite for marker '{marker}' in cell type '{cell_type}'. Skipping.")
                        continue

                    combined_marker_results[cell_type][marker] = {'z': combined_z, 'p': combined_p}

                except Exception as e:
                    print(f"Error combining p-values for marker '{marker}' in cell type '{cell_type}': {e}")
        
        return marker_expression_responder, marker_expression_non_responder, combined_marker_results
        
