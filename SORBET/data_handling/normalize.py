"""Functions to pre-process raw omics data using standard pre-processing approaches.

Primarily wraps a number of functions in scanpy.pp
"""
from typing import List
import numpy as np

from anndata import AnnData
import scanpy as sc

from .graph_model import OmicsGraph

def normalize_graph(tissue_graph: OmicsGraph, normalize_method: str, normalize_args: dict = dict()) -> None:
    """Normalize a single sample. OmicsGraph objects are modified in place. 

    Shallow wrapper around _normalize_[x] methods. Multiple normalization methods included including: 
        'total_count': _normalize_by_total_count
        'log_normalize': _normalize_by_log,
        'pca': _normalize_by_pca,
        'z-normalize': _normalize_by_variance,
        'to-range': _normalize_to_range

    Additional args may be passed in the normalize_args dictionary. See specific functions for options.
    
    Args:
        tissue_graph: an OmicsGraph object to normalize
        normalize_method: the chosen method to normalize a graph
        normalize_args: dictionary including additional arguments, specific to each method.

    """
    assert normalize_method in normalization_method_fns
  
    data = tissue_graph.get_node_data()
    normed_data = normalization_method_fns[normalize_method](data, normalize_method, **normalize_args) 
    tissue_graph.set_node_data(normed_data) 

def normalize_dataset(tissue_graphs: List[OmicsGraph], normalize_method: str, normalize_args: dict = dict()) -> None:
    """Normalizes across multiple samples. Shallow wrapper around _normalize_[x] methods.
    
    Can be helpful for methods that normalize across cells (e.g., PCA).

    Shallow wrapper around _normalize_[x] methods. Multiple normalization methods included including: 
        'total_count': _normalize_by_total_count
        'log_normalize': _normalize_by_log,
        'pca': _normalize_by_pca,
        'z-normalize': _normalize_by_variance,
        'to-range': _normalize_to_range

    Additional args may be passed in the normalize_args dictionary. See specific functions for options.
    
    Args:
        tissue_graph: an OmicsGraph object to normalize
        normalize_method: the chosen method to normalize a graph
        normalize_args: dictionary including additional arguments, specific to each method.
    """
    assert normalize_method in normalization_method_fns

    # Serialize data into single matrix:
    data, data_indices = list(), list()
    last_index = 0 
    for graph in tissue_graphs:
        graph_data = graph.get_node_data()
        
        sidx = last_index
        last_index = sidx + graph_data.shape[0]

        data_indices.append((sidx, last_index))
        data.append(graph_data)
    
    data = np.vstack(data)
    normed_data = normalization_method_fns[normalize_method](data, normalize_method, **normalize_args)
    
    # De-serialize data matrix:
    for tissue_graph, (sidx, eidx) in zip(tissue_graphs, data_indices):
        sample_normed_data = normed_data[sidx:eidx]
        tissue_graph.set_node_data(sample_normed_data)

def _normalize_by_total_count(data: np.ndarray, normalize_method: str, normalize_total_args: dict = dict()) -> np.ndarray:
    """Normalize each cell by total count. Wraps scanpy.pp.normalize_total

    Args:
        data: node data as a NumPy array
        normalize_method: either 'total_count' or 'log_normalize'.
            If 'log_normalize', scanpy.pp.log1p is applied first. 
        normalize_total_args: additional arguments passed as a dictionary. Options include:
                'exclude_highly_expressed', 'max_fraction', 'target_sum' 
            corresponding to scanpy.pp.normalize_total inputs. 

    Returns:
        Returns the data matrix with applied normalization.
    """

    if normalize_method ==  'log_normalize':
        data = sc.pp.log1p(data, copy=True)

    adata = AnnData(data)
    
    default_args = {
            'exclude_highly_expressed': True,
            'max_fraction': 0.05,
            'target_sum': 1.0
            }
    default_args.update(normalize_total_args)
    default_args['inplace'] = False 
   
    normed_data = sc.pp.normalize_total(adata, **default_args)['X']
    return normed_data

def _normalize_by_log(data: np.ndarray, normalize_method: str, normalize_log_args: dict = dict(), pseudocount: float = 1.0):
    """Normalize each cell by log(x+p). Wraps scanpy.pp.log1p, with an option for rescaling to total normalization

    Args:
        data: node data as a NumPy array
        normalize_method: 'log_normalize'. Not used.
        normalize_log_args: additional arguments passed as a dictionary.
            'rescale': if included in dictionary, normalize by total count.
        pseudocount: if not 1.0, normalize by different normalization factor

    Returns:
        Returns the data matrix with applied normalization.
    """

    if pseudocount == 1.0:
        normed_data = sc.pp.log1p(data, copy=True)
    else:
        data_cp = np.copy(data)
        normed_data = np.log(data_cp + pseudocount)

    if 'rescale' in normalize_log_args:
        normed_data = normed_data / normalize_log_args['rescale']
    
    return normed_data

def _normalize_by_pca(data: np.ndarray, normalize_method: str, pca_args: dict = dict(), fraction_variance_explained: float = 0.9): 
    """Normalize each cell by projecting onto principle components. Wraps scanpy.pp.pca

    Args:
        data: node data as a NumPy array
        normalize_method: 'pca'. Not used.
        pca_args: additional arguments passed to scanpy.pp.pca. See ScanPy for documentation. 
        fraction_variance_explained: ratio of explained variance to cut-off included principle components.

    Returns:
        Returns the data matrix with applied normalization.
    """

    adata = AnnData(data)

    default_args = {}
    default_args.update(pca_args)
    default_args['inplace'] = False

    pca_results = sc.pp.pca(adata, **default_args)
    var = pca_results.uns['pca']['variance_ratio']
    
    idx = np.min(np.argwhere(np.cumsum(var) >= fraction_variance_explained))
    normed_data = pca_results.obsm['X_pca'][:,:idx]

    return normed_data

def _normalize_by_variance(data: np.ndarray, normalize_method: str, zero_center: bool = True): 
    """Scales data to unit variance and zero mean. Wraps scanpy.pp.scale

    Args:
        data: node data as a NumPy array
        normalize_method: 'z-normalize'. Not used.
        zero_center: boolean encoding whether data are centered.

    Returns:
        Returns the data matrix with applied normalization.
    """

    return sc.pp.scale(data, zero_center = zero_center, copy = True)

def _normalize_to_range(data: np.ndarray, normalize_method: str):
    """Scales data to [0,1] range. Applied on each marker.

    For example, a measured marker, x_i would receive:
        (x_i - x_min) / (x_max - x_min)

    Args:
        data: node data as a NumPy array
        normalize_method: 'to-range'. Not used.

    Returns:
        Returns the data matrix with applied normalization.
    """


    mi, ma = np.min(data, axis=0), np.max(data, axis=0)
    return (data - mi) / (ma - mi)

normalization_method_fns = {
        'total_count': _normalize_by_total_count,
        'log_normalize': _normalize_by_log,
        'pca': _normalize_by_pca,
        'z-normalize': _normalize_by_variance,
        'to-range': _normalize_to_range
        }

def normalize_graphs_by_pca(tissue_graphs: List[OmicsGraph], pca_args: dict = dict()):
    """Deprecated: To remove on future versions.
    """

    combined_data = list()
    offsets, cidx = [0], 0
    
    for graph in tissue_graphs:
        graph_data = graph.get_node_data()
        combined_data.append(graph_data)

        cidx += graph_data.shape[0]
        offsets.append(cidx)
    
    combined_data = np.vstack(combined_data)
    adata = AnnData(combined_data)

    default_args = {}
    default_args.update(pca_args)
    default_args['copy'] = True

    pca_results = sc.pp.pca(adata, **default_args)
    pca_data = pca_results.obsm['X_pca']
    
    mupd = ['PC {idx}' for idx in range(pca_data.shape[1])]
    for sidx, eidx, tissue_graph in zip(offsets, offsets[1:], tissue_graphs):
        normed_data = pca_data[sidx:eidx]
        tissue_graph.set_node_data(normed_data, marker_update = mupd) 
