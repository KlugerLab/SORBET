"""Functions for converting full samples (as OmicsGraphs) into subgraphs.

Files are first converted in OmicsGraph objects representing subgraphs using the create_subgraphs
function. Then, they are converted to torch objects (for use with learning models) using the 
create_torch_subgraphs function.

Files are stored in structured directories with a chosen filepath. If `subgraph_data/` is the passed filename,
the files are stored in the following structure:
```
subgraph_data/
    - graphs_py/ # Contains OmicsGraph subgraphs
        - {graph_id_a}_sg_1.p
        - {graph_id_a}_sg_2.p
        ...
        - {graph_id_b}_sg_1.p
        - {graph_id_b}_sg_2.p
        ...
    - graphs_torch/ # Contains torch subgraphs
        - {graph_id_a}_sg_1.p
        - {graph_id_a}_sg_2.p
        ...
        - {graph_id_b}_sg_1.p
        - {graph_id_b}_sg_2.p
        ...
    - py_subgraph_mapping.csv # Mapping from input OmicsGraph files to files in associated subgraph files in graphs_py
    - torch_subgraph_mapping.csv # Mapping from subgraph files in graphs_py/ to files in graphs_torch/
```
{graph_id_a} corresponds to the graph IDs of full OmicsGraph objects. The `_sg_{x}` identifier
appended to the end of each file denotes an (arbitrarily-indexed) subgraph extracted from {graph_id_a}.

Note: These values can be changed in the relevant functions. We discourage changing the values.
"""
import os, sys
import csv

import numpy as np
import torch
from torch_geometric.data import Data

from .graph_model import OmicsGraph, load_omicsgraph, dump_omicsgraph
from .subgraph_extraction import subgraph_extraction

def create_subgraphs(complete_graphs_dirpath: str, output_dirpath: str, subgraph_extraction_algorithm: str, subgraph_extraction_algorithm_kwargs: dict, 
        mapping_fname: str = "py_subgraph_mapping.csv", subgraph_dirname: str = "graphs_py", subgraph_extraction_type: str = "arbitrary") -> None:
    """Converts OmicsGraph objects into subgraphs using a specified subgraph extraction algorithm.

    Functions as a file-to-file conversion. Files are input as pickled OmicsGraph objects and output
    to pickled OmicsGraph objects in a specified output directory.

    Should be combined with create_torch_subgraphs (see below) for creating data inputs to SORBET. 

    Args:
        complete_graphs_dirpath: directory containing the input OmicsGraph files (pickled)
        output_dirpath: a directory path where output objects are stored
        subgraph_extraction_algorithm: the chosen algorithm for extracting subgraphs (see subgraph_extraction.py)
        subgraph_extraction_algorithm_kwargs: additional arguments passed to the subgraph extraction algorithms
        mapping_fname: a file name, placed in [output_dirpath] mapping input file paths to output subgraph file paths. Should not change.
        subgraph_dirname: the directory where OmicsGraph objects are output to. Placed at [output_dirpath]/[subgraph_dirname]. Should not change.
        subgraph_extraction_type: [Deprecated] Ignore.
    """

    subgraph_file_mapping = list()
    
    pygraphs_output_dirpath = os.path.join(output_dirpath, subgraph_dirname)
    if not os.path.exists(pygraphs_output_dirpath):
        os.makedirs(pygraphs_output_dirpath)

    for ifile in os.listdir(complete_graphs_dirpath):
        fpath = os.path.join(complete_graphs_dirpath, ifile)
        graph = load_omicsgraph(fpath)

        graph_id = os.path.splitext(ifile)[0]
        
        subgraph_files = list()
        
        subgraphs = subgraph_extraction(graph, subgraph_extraction_algorithm, subgraph_extraction_algorithm_kwargs)
        for sg_idx, sg in enumerate(subgraphs):
            fname = f'{graph_id}_sg_{sg_idx}.p'
            ofpath = os.path.join(pygraphs_output_dirpath, fname)
            dump_omicsgraph(sg, ofpath)
    
            subgraph_files.append(os.path.join(subgraph_dirname, fname))
            
        subgraph_file_mapping.append([graph_id, fpath, *subgraph_files])
    
    mapping_fpath = os.path.join(output_dirpath, mapping_fname)
    with open(mapping_fpath, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Graph_ID", "Graph_File", "Subgraph_Files(multicol)"])
        writer.writerows(subgraph_file_mapping)

def _create_torch_subgraph(igraph: OmicsGraph, ofpath: str) -> None:
    """Helper function creating (and saving) a torch graph object from an OmicsGraph input.

    Args:
        igraph: input OmicsGraph object
        ofpath: output filepath where the resulting torch graph is saved.
    """
    node_features = np.zeros((len(igraph.vertices), len(igraph.markers)), dtype=np.float64)
    for vidx, vertex in enumerate(igraph.vertices):
        marker_data = np.array([igraph.node_attributes[vertex][mi] for mi in igraph.markers])
        node_features[vidx] = marker_data

    vertex_mapping = {vertex:vidx for vidx, vertex in enumerate(igraph.vertices)}
    
    edges = set(igraph.graph.edges())
    edge_index = np.zeros((2, len(edges) * 2), dtype=int)
    for eidx, (ei, ej) in enumerate(igraph.graph.edges()):
        ei_m, ej_m = vertex_mapping[ei], vertex_mapping[ej]
        edge_index[:, 2 * eidx] = [ei_m, ej_m]
        edge_index[:, 2 * eidx + 1] = [ej_m, ei_m]
    
    label = igraph.graph_label

    subgraph = Data(x = torch.tensor(node_features, dtype=torch.float), 
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            y = torch.tensor([label], dtype=torch.long))
    torch.save(subgraph, ofpath)

def create_torch_subgraphs(output_dirpath: str, mapping_fname: str = "torch_subgraph_mapping.csv", 
        py_subgraphs_dirname: str = "graphs_py", torch_subgraphs_dirname: str = "graphs_torch") -> None:
    """Converts OmicsGraph subgraphs, stored in a single directory, into torch subgraphs.

    Follows create_subgraphs (and assumes the defined file structure).  

    Args:
        output_dirpath: the processed directory, as created using the create_subgraphs function
        mapping_fname: a filename for a mapping between input subgraph files and output torch files. Should not be changed.
        py_subgraphs_dirname: directory name, appended to output_dirpath, where input OmicsGraph files are located. Should not be changed.
        torch_subgraphs_dirname: directory name, appended to output_dirpath, where output torch graphs are stored. Should not be changed.
    """
    file_mapping = list()
    torchgraphs_output_dirpath = os.path.join(output_dirpath, torch_subgraphs_dirname)
    if not os.path.exists(torchgraphs_output_dirpath):
        os.makedirs(torchgraphs_output_dirpath)
    
    py_subgraphs_dirpath = os.path.join(output_dirpath, py_subgraphs_dirname)
    for ifile in os.listdir(py_subgraphs_dirpath):
        input_omicsgraph_fpath = os.path.join(py_subgraphs_dirpath, ifile)
        input_omicsgraph = load_omicsgraph(input_omicsgraph_fpath)
        
        output_fname = f'{os.path.splitext(ifile)[0]}.pt'
        output_torchgraph_fpath = os.path.join(torchgraphs_output_dirpath, output_fname)
        _create_torch_subgraph(input_omicsgraph, output_torchgraph_fpath)
        
        rel_pygraph_fpath = os.path.join(py_subgraphs_dirname, ifile)
        rel_torch_fpath = os.path.join(torch_subgraphs_dirname, output_fname)
        file_mapping.append([rel_pygraph_fpath, rel_torch_fpath])
    
    mapping_fpath = os.path.join(output_dirpath, mapping_fname)
    with open(mapping_fpath, 'w+') as ofile:
        writer = csv.writer(ofile, delimiter=',')
        writer.writerow(["Input_OmicsGraph", "Output_TorchGraph"])
        writer.writerows(file_mapping)
