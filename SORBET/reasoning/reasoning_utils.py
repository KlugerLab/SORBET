from typing import List, Tuple
import os

import numpy as np

import scanpy as sc
from anndata import AnnData

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from ..data_handling import load_omicsgraph
from ..learning import GCNSorbetBase 

# TODO: load_model_cell_embeddings and load_model_subgraph_embeddings currently compute embeddings and load them.
# This is somewhat of a misnomer. Consider converting name to a different choice (e.g., compute_[x])

def load_model_cell_embeddings(model: GCNSorbetBase, dataset: Dataset, data_dir: str, ofpath: str = None, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes cell embeddings for all cells across all samples

    Args:
        model: input SORBET model for downstream analysis
        dataset: the torch geometric dataset used for  
        data_dir: the path to the folder containing the OmicsGraph datasets 
        ofpath: optional output filepath to save returned values to
        device: device to use for prediction
        batch_size: batch sizes used in training / evaluating models

    Returns: 
        Four arrays of equal first dimension (number of vertices, n) including:
            An (n x d) cell embedding of the data of dimension d
            A length n arrray of the subgraph labels
            A length n array of strings indicating the associated subgraph 
            A length n array of the vertex indices, corresponding to the indexing in the input subgraphs.
    """
    embeddings = list()
    labels = list()
    subgraphs = list()
    vertices = list()

    model.to(device)
    with torch.no_grad():
        model.eval()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        tot = 0

        for data, sgs in zip(dataloader, _grouper(dataset.processed_file_names, batch_size)):
            data.to(device)
            
            tot += data.x.shape[0]

            embedding = model.get_cell_embedding(data.x, data.edge_index, data.batch)
            embeddings.append(embedding.cpu().numpy())
            
            batch_idxes = data.batch.cpu().numpy()
            _labels = data.y.cpu().numpy()
            for bi, li, fname in zip(sorted(set(batch_idxes)), _labels, sgs): 
                ncells = np.count_nonzero(batch_idxes == bi)
                
                labels.extend(li for _ in range(ncells))
                sg_name = os.path.splitext(os.path.basename(fname))[0]
                subgraphs.extend(sg_name for _ in range(ncells))
                
                omics_graph = load_omicsgraph(os.path.join(data_dir, "graphs_py", f'{sg_name}.p'))
                vertices.extend(omics_graph.vertices)

                assert len(omics_graph.vertices) == ncells
            
            assert len(set(batch_idxes)) == len(sgs)
        

    embeddings = np.vstack(embeddings)
    labels, vertices = np.array(labels), np.array(vertices)

    if ofpath is not None:
        np.savez(ofpath, embeddings=embeddings, labels=labels, subgraphs=subgraphs, vertices=vertices)

    return embeddings, labels, subgraphs, vertices 

def load_model_subgraph_embeddings(model: GCNSorbetBase, dataset: Dataset, ofpath: str = None, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes sub-graph embeddings for all sub-graph samples

    Args:
        model: input SORBET model for downstream analysis
        dataset: the torch geometric dataset used for  
        data_dir: the path to the folder containing the OmicsGraph datasets 
        ofpath: optional output filepath to save returned values to
        device: device to use for prediction
        batch_size: batch sizes used in training / evaluating models

    Returns: 
        Three arrays of equal first dimension (number of vertices, n) including:
            An (n x d) subgraph embedding of the data of dimension d
            A length n arrray of the subgraph labels
            A length n array of strings indicating the associated subgraph 
    """
    embeddings = list()
    labels = list()
    subgraphs = dataset.processed_file_names 
    subgraphs = [os.path.splitext(os.path.basename(sg))[0] for sg in subgraphs]

    model.to(device)
    with torch.no_grad():
        model.eval()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for data in dataloader:
            data.to(device)
            
            ce = model.get_cell_embedding(data.x, data.edge_index, data.batch)
            embedding = model.get_subgraph_embedding(ce, data.edge_index, data.batch)
            embeddings.append(embedding.cpu().numpy())
            
            label = data.y.cpu().numpy()
            labels.extend(data.y.cpu().numpy().flatten())
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    if ofpath is not None:
        np.savez(ofpath, embeddings=embeddings, labels=labels, subgraphs=subgraphs)

    return embeddings, labels, subgraphs 

def load_model_predictions(model: GCNSorbetBase, dataset: Dataset, ofpath: str = None, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Computes sub-graph embeddings for all sub-graph samples

    Args:
        model: input SORBET model for downstream analysis
        dataset: the torch geometric dataset used for  
        ofpath: optional output filepath to save returned values to
        device: device to use for prediction
        batch_size: batch sizes used in training / evaluating models

    Returns: 
        Two arrays of equal first dimension (number of vertices, n) including:
            A length n array of predictions for each subgraph 
            A length n arrray of the subgraph labels

    """
    predictions = list()
    subgraphs = dataset.processed_file_names 
    subgraphs = [os.path.splitext(os.path.basename(sg))[0] for sg in subgraphs]

    model.to(device)
    with torch.no_grad():
        model.eval()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for data in dataloader:
            data.to(device)
            
            preds = model.predict(data.x, data.edge_index, data.batch)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions), subgraphs 

def load_model_precomputed_embedding(ofpath: str):
    """Helper function loading pre-computed cell embeddings. 
    Can be either subgraph or cell embeddings.

    Requires running embedding extraction functions above: load_model_cell_embeddings, load_model_subgraph_embeddings.

    Args:
        ofpath: filepath to saved embeddings

    Returns:
        Embeddings with the same structure as in the two functions above. 
    """
    npzf = np.load(ofpath)
    if "vertices" in npzf.keys():
        return npzf['embeddings'], npzf['labels'], npzf['subgraphs'], npzf['vertices']

    return npzf['embeddings'], npzf['labels'], npzf['subgraphs']

def compute_cell_clustering(cell_data: np.ndarray, n_neighbors: int = 10, n_pcs: int = 40, leiden_resolution: int = 1, 
        subgraph_labels: List[str] = None, cell_ids: List[int] = None, ofpath: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes cell clustering via Leiden clustering algorithm 
    
    Args:
        cell_data: an (n x g) data encoding the cell profile of each data point
        n_neighbors: number of neighbors computed in pre-proessing to computer Leiden (see scanpy.pp.neighbors)
        n_pcs: number of principle components for pre-processing data
        leiden_resolution: Leiden resolution hyperparameter (see scanpy.tl.leiden)
        subgraph_labels: length n list of subgraphs for each data point
        cell_ids: length n list of cell indices of each data point
        ofpath: optional output filepath to save returned values to

    Returns: 
        Three arrays of equal first dimension (number of vertices, n) including:
            A length n arrray of the subgraph labels
            A length n array of the cell indices 
            A length n array of the assigned cluster index. 
    """
    adata = AnnData(cell_data)
    sc.pp.neighbors(adata, n_neighbors = n_neighbors, n_pcs = n_pcs)
    sc.tl.leiden(adata, resolution = leiden_resolution)
    
    clust_ids = adata.obs['leiden'].to_numpy().astype(int)

    if ofpath is not None:
        np.savez(ofpath, subgraph_labels = subgraph_labels, cell_ids = cell_ids, clustering = clust_ids)

    return subgraph_labels, cell_ids, clust_ids

def load_cell_clustering(ofpath: str):
    """Helper function loading pre-computed cell clustering

    Args:
        ofpath: filepath to saved embeddings

    Returns:
        Embeddings with the same structure as in compute_cell_clustering. 
    """
    npzf = np.load(ofpath)
    return npzf['subgraph_labels'], npzf['cell_ids'], npzf['clustering']

def _grouper(iterable, n, fillvalue=None):
    """A helper function iterating over an iterable 
    """
    chunk = list()
    for i in iterable:
        chunk.append(i)
        if len(chunk) == n:
            yield chunk
            chunk = list()
    if len(chunk) != 0:
        yield chunk
