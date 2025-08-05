import os
import re
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import CCA

from ..data_handling import load_omicsgraph
from .cca_model import L0SparseCCA
from .cca_model import sparse_cca_kfold_cv 

_default_lambda = 1
_default_lambdas = np.logspace(-6, -1, 6) 

@dataclass
class CCADataIndexing:
    row_indexing: np.ndarray
    graph_indexing: Dict[int, str]
    subgraph_indexing: np.ndarray 
    in_col_indexing: list

@dataclass
class CCAData:
    in_array: np.ndarray
    out_array: np.ndarray
    indexing: CCADataIndexing

#Note: what are the Acronyms of CCAD
def preprocess_graph_structured_data(input_data_dir: str, output_data: np.ndarray, output_labels: list, output_vertex_indices: list, 
        nhops: Optional[int] = 1, exact_hops: Optional[bool] = True) -> CCAData:
    """Creates a CCAData object from input data corresponding to graph-aware CCA. 
    Pairs all input data values with the appropriate graph-structured pairings.

    Args:
        input_data_dir: a data directory including OmicsGraph graphs; typically, extracted subgraphs.
        output_data: an (n x d) array of each vertex's embedding 
        output_labels: a length n list of graph labels
        output_vertex_indices: a length n list of vertex indices
        nhops: number of hops to consider. Default: 1.
        exact_hops: whether to treat hop distance (nhops) as exact or include distances less than nhops. Default: True

    Returns:
        CCAData object mapping input data array to output data array.
    """
    graph_regex = r'(.*)_sg_\d+(\.p)?'
    subgraph_index_regex = r'.*_sg_(\d+)(\.p)?'
    
    output_graph_labels = np.array(list(map(lambda gi: re.match(graph_regex, gi).group(1), output_labels)))
    output_subgraph_labels = np.array(list(map(lambda gi: int(re.match(subgraph_index_regex, gi).group(1)), output_labels)))

    in_array = list()
    out_array = list()

    graph_indexing, graph_index_ctr = dict(), -1
    subgraph_indexing, row_indexing = list(), list()
    marker_labels = None
    for ifile in os.listdir(input_data_dir):
        graph_label = re.match(graph_regex, ifile).group(1)
        subgraph_index = int(re.match(subgraph_index_regex, ifile).group(1))

        if graph_label not in graph_indexing:
            graph_index_ctr += 1
            graph_indexing[graph_label] = graph_index_ctr
        
        graph_index = graph_indexing[graph_label]

        graph = load_omicsgraph(os.path.join(input_data_dir, ifile))
        node_data, vertex_order = graph.get_node_data(), graph.vertices
        vertex_indexing = {vi:idx for idx, vi in enumerate(graph.vertices)}
        
        _source_data = output_data[np.logical_and(output_graph_labels == graph_label, output_subgraph_labels == subgraph_index)] 
        
        for vertex in vertex_order: 
            neighbors = set(graph.get_khop_neighborhood(vertex, nhops))
            if exact_hops and nhops > 1:
                k_minus_1_neighbors = graph.get_khop_neighborhood(vertex, nhops - 1)
                neighbors  = neighbors.difference(k_minus_1_neighbors)
            neighbors.discard(vertex) # Ensures neighbor is not near itself.
            
            # Update Data:
            in_array.extend(node_data[vertex_indexing[vi]] for vi in neighbors) 
            vertex_source_data = _source_data[vertex_indexing[vertex]]
            out_array.extend(vertex_source_data for _ in range(len(neighbors)))
            
            # Update indexing:
            row_indexing.extend([vi, vertex, graph_index, subgraph_index] for vi in neighbors)

        if marker_labels is None:
            marker_labels = graph.markers
    
    indexing = CCADataIndexing(
                row_indexing = np.array(row_indexing),
                graph_indexing = graph_indexing,
                subgraph_indexing = np.array(subgraph_indexing),
                in_col_indexing = marker_labels
            )

    return CCAData(
                in_array = np.vstack(in_array),
                out_array = np.vstack(out_array),
                indexing = indexing
            )

def dump_cca_data(data: CCAData, ofile: str):
    """Save a CCAData Object. Formatted in a numpy array (.npz)

    Args:
        data: a CCAData object to save
        ofile: output filepath.
    """
    output_data = {
            "in_array": data.in_array,
            "out_array": data.out_array
            }
    output_data.update(_dump_indexing(data.indexing))
    
    np.savez(ofile, **output_data)

def load_cca_data(ifile: str) -> CCAData:
    """Returns a previously saved CCAData object

    Args:
        ifile: input filepath.

    Returns:
        Saved CCAData Objecct
    """
    data_file = np.load(ifile)
    
    indexing = _load_indexing(data_file)
    return CCADataIndexing(
                in_array = data_file['in_array'],
                out_array = data_file['out_array'],
                indexing = indexing
            )

def _dump_indexing(indexing: CCADataIndexing) -> Dict[str, np.ndarray]:
    """Helper function to save indexing data to .npz file.
    Called by dump_cca_data function above.

    Args:
        indexing: a CCADataIndexing object

    Returns:
        A dictionary serializing the CCADataIndexing object
    """
    indexing_dct = {
            "row_indexing": indexing.row_indexing,
            "in_col_indexing": np.array(indexing.in_col_indexing),
            "subgraph_indexing": indexing.subgraph_indexing
            }
    
    _k, _v = map(list, *indexing.graph_indexing) 
    indexing_dct.update({"graph_indexing_keys": _k, "graph_indexing_values": _v})

    return indexing_dct

def _load_indexing(npz_filehandle) -> CCADataIndexing:
    """Helper function to lload indexing data from .npz file

    Args:
        npz_filehandle: a loaded numpy array serialized in the format of _dump_indexing

    Returns:
        A CCADataIndexing object.
    """
    return {
            "row_indexing": npz_filehandle['row_indexing'],
            "graph_indexing": {int(ki):str(vi) for ki, vi in zip(npz_filehandle['graph_indexing_keys'], npz_filehandle['graph_indexing_values'])}, 
            "subgraph_indexing": npz_filehandle['subgraph_indexing'], 
            "in_col_indexing": npz_filehandle['in_col_indexing'].tolist()
            }
