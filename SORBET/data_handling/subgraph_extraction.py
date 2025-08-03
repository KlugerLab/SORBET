"""Algorithms for subgraph exctraction. Primary algorithm implemented in _microenvironment_subgraph_extraction.
"""
from typing import List
import numpy as np

from .graph_model import OmicsGraph

def subgraph_extraction(tissue_graph: OmicsGraph, extraction_method: str, extraction_args: dict = dict()) -> List[OmicsGraph]:
    """Implements a basic interface to access different subgraph extraction methods (below)


    Options for subgraph extaction include:
        "microenvironment": _microenvironment_subgraph_extraction,
        "arbitrary": _arbitrary_subgraph_extraction,
        "heat_diffusion": _heat_diffusion_subgraph_extraction
    Heat diffusion extraction is not implemented.

    Args:
        tissue_graph: OmicsGraph object, which the subgraph extraction algorithm will be applied on
        extraction_method: chosen method. Options defined above.
        extraction_args: dictionary with additional arguments passed to the subgraph extraction algorithm
    
    Returns:
        A list of OmicsGraph objects representing the extracted subgraphs. 
    """
    assert extraction_method in subgraph_extraction_methods
    return subgraph_extraction_methods[extraction_method](tissue_graph, **extraction_args)

def _microenvironment_subgraph_extraction(tissue_graph: OmicsGraph, marker: str, k: int, minimum_size: int) -> List[OmicsGraph]:
    """Subgraph extraction algorithm prioritzing by expression of a chosen marker.

    Args:
        tissue_graph: input OmicsGraph object
        marker: chosen marker to prioritize
        k: expansion size around each cell (i.e., neighborhood size)
        minimize_size: minimum size of extracte subgraphs (in number of cells) 

    Returns:
        A list of OmicsGraph objects representing the extracted subgraphs fitting the defined parameters.
    """
    subgraphs = list()
    
    vertices, marker_values = tissue_graph.get_marker(marker)
    T = np.median(marker_values)
    Q = [vertices[i] for i in np.argsort(marker_values)]

    m = Q.pop()
    while tissue_graph.get_marker(marker, [m])[1][0] >= T: # TODO: Very ugly. Fix.
        neighborhood_1 = list(tissue_graph.get_khop_neighborhood(m, 1))
        _, marker_values = tissue_graph.get_marker(marker, neighborhood_1)
        
        if np.median(marker_values) >= T:
            neighborhood_k = list(tissue_graph.get_khop_neighborhood(m, k))
            
            if len(neighborhood_k) > minimum_size:
                neighborhood_k.append(m)
                
                subgraph = tissue_graph.make_subgraph(neighborhood_k)
                subgraphs.append(subgraph)

                Q = [vi for vi in Q if vi not in neighborhood_k]

        if len(Q) == 0: break
        
        m = Q.pop()
    
    return subgraphs

def _heat_diffusion_subgraph_extraction(tissue_graph: OmicsGraph, marker: str, k: int, minimum_size: int):
    """Subgraph extraction via heat diffusion-like process. Not implemented. 
    """
    # TODO: Extract subgraphs using heat-diffusion like graph cover

    return [tissue_graph]

def _arbitrary_subgraph_extraction(tissue_graph: OmicsGraph, marker: str, k: int, minimum_size: int):
    """Subgraph extraction via an arbitrary selection of (minimally-overlapping) subgraphs.

    Nodes are chosen in vertex order.

    Args:
        tissue_graph: input OmicsGraph object
        marker: chosen marker to prioritize
        k: expansion size around each cell (i.e., neighborhood size)
        minimize_size: minimum size of extracte subgraphs (in number of cells) 

    Returns:
        A list of OmicsGraph objects representing the extracted subgraphs fitting the defined parameters.
    """

    subgraphs = list()
    vertices, marker_values = tissue_graph.get_marker(marker)
    Q = list(vertices)
    seen = set()
    
    pop_item = lambda v, Q: [vi for vi in Q if vi != v]

    while len(Q) > 0: 
        Vi = Q[np.random.choice(np.arange(len(Q)))]
        Q = pop_item(Vi, Q)
        
        if len(seen.intersection(tissue_graph.get_khop_neighborhood(Vi, 1))) != 0:
            continue

        neighborhood_k = list(tissue_graph.get_khop_neighborhood(Vi, k))
        if len(neighborhood_k) >= minimum_size: 
            seen |= set(neighborhood_k)
            
            subgraph = tissue_graph.make_subgraph(neighborhood_k)
            subgraphs.append(subgraph)

    return subgraphs


subgraph_extraction_methods = {
        "microenvironment": _microenvironment_subgraph_extraction,
        "arbitrary": _arbitrary_subgraph_extraction,
        "heat_diffusion": _heat_diffusion_subgraph_extraction
        }
