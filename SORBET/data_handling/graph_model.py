"""A data class encoding a spatial 'omics sample in an object with useful helper functions.
"""
from dataclasses import dataclass
from typing import Union
import pickle
import numpy as np
import networkx as nx

class OmicsGraph:
    """
    Base class encoding a spatial graph representation of a profiled sample. 
    
    Facilitates two major graph operations: get_khop_neighborhood and make_subgraph. 
        get_khop_neighborhood: a function to extract the k-hop neighborhood around a chosen node
        make_subgraph: makes a subgraph out of a specified list of nodes

    Attributes:
        graph: a networkx graph encoding the spatial relationships of cells
        node_attributes: a dictionary mapping nodes to marker values
        markers: markers profiled in the sample
        vertices: list of vertices (or nodes) in the sample.
        graph_label: integer encoding the attributed label of the graph
        meta_markers: list of markers associated with the cell not used in core inference.
            An example would be proteins measured simultaneously with transcriptomics data.
        meta_markers_data: dictionary mapping vertices to meta markers (shared order with meta_markers) 
        node_meta_attributes: parallels the node_attributes structure for the meta_marker subset
    """

    def __init__(self, vertex_lst: list, edge_lst: list, data: np.ndarray, markers: list, graph_label: int, meta_markers: Union[list, dict] = None):
        """Intializes the OmicsGraph object.

        Args:
            vertex_lst: list of vertices included in the sample
            edge_lst: list of vertex tuples encoding (graph) relationships between vertices 
            data: NumPy array encoding (vertex_lst) x (markers) data 
            markers: list of markers measured in the sample
            graph_label: integer encoding the relevant value associated with the
            meta_markers: list of meta-markers and dictionary of vertex association, as described in the class definition 
        """
        self.graph = nx.Graph()
        
        self.node_attributes = dict()
        for vi, vertex in enumerate(vertex_lst):
            self.graph.add_node(vertex) 
            
            marker_mapping = {marker:value for marker, value in zip(markers, data[vi])}
            self.node_attributes[vertex] = marker_mapping

        self.graph.add_edges_from(edge_lst)
        
        self.markers = markers
        self.vertices = vertex_lst
        self.graph_label = graph_label

        if meta_markers is not None:
            self.meta_markers = meta_markers[0]
            self.meta_marker_data = meta_markers[1]

            self.node_meta_attributes = dict()
            for vi, vertex in enumerate(vertex_lst):
                meta_marker_mapping = {marker:value for marker, value in zip(self.meta_markers, self.meta_marker_data[vi])} 
                self.node_meta_attributes[vertex] = meta_marker_mapping
        else:
            self.meta_markers = None
            self.meta_marker_data = None
            self.node_meta_attributes = None

    def get_marker(self, marker: str, nodes: list = None) -> np.ndarray:
        """Getter method to extract the marker expression for a single marker.

        Can be either across all nodes or for a specified subset of nodes.

        Args:
            marker: a chosen to extract data
            nodes: an optional list of nodes for which data is extracted

        Returns:
            A NumPy array of the specified marker expression across either (a) all nodes
            or (b) a subset of nodes. Data are returned either in originally defined order
            (vertex_lst, (a)) or the passed orders of nodes (b). 
        """

        if marker in self.markers:
            attrs = self.node_attributes
        else:
            assert self.meta_markers is not None and marker in self.meta_markers
            attrs = self.node_meta_attributes

        if nodes is not None:
            marker_vals = [attrs[vi][marker] for vi in nodes]
            return nodes, marker_vals 
        else:
            marker_vals = [attrs[vi][marker] for vi in self.vertices]
            return self.vertices, marker_vals
    
    def get_khop_neighborhood(self, vertex: int, k: int) -> list:
        """Extracts a k-hop neighborhood around a chosen vertex.

        Args:
            vertex: chosen vertex around which a neighborhood is extracted
            k: chosen neighborhood size

        Returns:
            A list of neighboring vertices in the k-hop neighborhood of the defined vertex. 
        """
        neighbors = set(self.graph.neighbors(vertex))
        
        new_neighbors = set(neighbors)
        for kh in range(k - 1):
            if len(new_neighbors) == 0: break

            next_hop = set.union(*(set(self.graph.neighbors(vi)) for vi in new_neighbors))
            new_neighbors = next_hop.difference(neighbors)
            neighbors |= next_hop 

        return list(neighbors)
    
    def make_subgraph(self, vertex_lst: list) -> OmicsGraph:
        """Makes a subgraph from the subset of vertices included 

        Args:
            vertex_lst: a list of vertices that define the desired subgraph.

        Returns:
            A new OmicsGraph object defined on the subset of passed vertices.
        """
        subgraph = self.graph.subgraph(vertex_lst)
        
        V = list(subgraph.nodes())
        E = list(subgraph.edges())
        
        X = list() 
        for vi in V:
            X.append([self.node_attributes[vi][marker] for marker in self.markers])
        X = np.array(X)
        
        if self.meta_markers is not None:
            M = list()
            for vi in V:
                M.append([self.node_meta_attributes[vi][marker] for marker in self.meta_markers])
            meta_marker_data = (self.meta_markers, np.array(M))
        else:
            meta_marker_data = None

        return OmicsGraph(V, E, X, self.markers, self.graph_label, meta_marker_data)

    def get_node_data(self) -> np.ndarray:
        """Getter method to extract node data for all vertices in the sample.

        Returns:
            A NumPy array returning all of the node data associated with the graph
        """
        nodes_data = list()
        for vertex in self.vertices:
            data_arr = [self.node_attributes[vertex][marker] for marker in self.markers]
            nodes_data.append(data_arr)
        
        return np.array(nodes_data)

    def set_node_data(self, node_data: np.ndarray, marker_update: list = None) -> None:
        """Setter method to re-set node data. Can be used to change the number of markers

        Useful for methods that, for example, change the normalization of nodes in a graph.

        Args:
            node_data: updated data of form (vertex_list) x (marker_lst | marker_update)
            marker_update: passed if the node_data includes a different set of markers than initially defined
        """
        assert len(self.vertices) == node_data.shape[0]
        if marker_update is not None:
            self.markers = marker_update

        for vertex, arr in zip(self.vertices, node_data):
            marker_mapping = {marker:val for marker, val in zip(self.markers, arr)}
            self.node_attributes[vertex] = marker_mapping 

def load_omicsgraph(fpath: str) -> OmicsGraph:
    """Loads an OmicsGraph object saved to the specified filepath. 

    To save an OmicsGraph, see dump_omicsgraph.

    Args:
        fpath: a file path where the object is saved (as a pickle object)

    Returns:
        The OmicsGraph object saved at fpath. 
    """
    with open(fpath, 'rb') as ifile:
        idata = pickle.load(ifile)
        
        vertex_lst = idata[0][1]
        edge_lst = idata[1][1]
        marker_lst = idata[2][1]
        marker_data = idata[3][1]
        graph_label = idata[4][1]
        
        meta_marker_data = idata[5][1] 

    return OmicsGraph(vertex_lst, edge_lst, marker_data, marker_lst, graph_label, meta_marker_data)

def dump_omicsgraph(input_graph: OmicsGraph, fpath: str) -> None:
    """Saves an OmicsGraph object to the specified filepath. 

    Objects are de-serialized into a pickle object. To reload, see load_omicsgraph.

    Args:
        input_graph: an OmicsGraph object to be saved
        fpath: a file path where the object is saved (as a pickle object)
    """
    vertex_lst = input_graph.vertices
    marker_lst = input_graph.markers
    edge_lst = list(input_graph.graph.edges())

    data = [[input_graph.node_attributes[vi][mj] for mj in marker_lst] for vi in vertex_lst]
    
    graph_label = input_graph.graph_label
    
    if input_graph.meta_markers is not None:
        meta_marker_data = [input_graph.meta_markers, input_graph.meta_marker_data]
    else:
        meta_marker_data = None

    with open(fpath, 'wb+') as ofile:
        odata = [
            ["vertices", vertex_lst],
            ["edges", edge_lst],
            ["markers", marker_lst],
            ["marker_data", data],
            ["graph_label", graph_label],
            ["meta_markers", meta_marker_data] 
        ]

        pickle.dump(odata, ofile)
