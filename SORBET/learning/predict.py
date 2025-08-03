from typing import List, Tuple, Any
import os
import re
import csv
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader

from .dataset import TorchOmicsDataset 
from .experiment_management import load_data_split_record 
from .train_utils import load_model 
from .models import BaseGraphModel 
from .train import _test_step

_output_combined_subgraph_fname = "subgraphs_combined.csv"
_output_combined_graph_fname = "graphs_combined.csv"

def predict_subgraphs(input_dirpath: str, root_fpath: str, metadata_files: str, model_type: BaseGraphModel,
        batch_size: int = 128, output_fname: str = _output_combined_subgraph_fname,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> List[Tuple[int, float, str]]:
    """Predict all subgraphs over a single experiment / data splits
    
    Args:
        input_dirpath: data dirpath of a single data split
        root_fpath: the root of the data directories
        metadata_files: a SubgraphMetadata data class defining the data structure in root_fpath 
        model_type: model class that optimization is applied to. See models/ 
        batch_size: batch size for training
        output_fname: output filename for predictions
        device: device to train model on

    Returns:
        List of predictions for each subgraph.
    """
    data_split, record = load_data_split_record(input_dirpath) 
    
    predictions, labels, subgraph_descriptors = list(), list(), list()

    for idx, spl in enumerate(data_split):
        model_prefix = os.path.join(record.models_dir, f'split_{idx}')
        model = load_model(model_type, model_prefix)
        model.to(device) 
        
        test_ds = TorchOmicsDataset(root_fpath, metadata_files, spl[-1])

        fpaths = test_ds.processed_file_names
        subgraph_descriptors.extend(_process_subgraph_ids(fpaths))
        
        test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=batch_size)
        _preds, _labs = _test_step(model, test_dataloader, device)
        predictions.extend(_preds)
        labels.extend(_labs)
    
    print(roc_auc_score(labels, predictions))
    
    combined_data = [[li, pi, *sgi] for li, pi, sgi in zip(labels, predictions, subgraph_descriptors)]

    header = ["Label", "Predictions", "Subgraph_ID", "Tissue_ID", "File_Name"]
    ofpath = os.path.join(input_dirpath, output_fname)
    with open(ofpath, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(combined_data)
    
    return combined_data

def export_split_subgraph_predictions(input_dirpath: str, split_idx: int,
        labels: np.ndarray, predictions: np.ndarray, test_dataset: TorchOmicsDataset,
        output_fname: str = _output_combined_subgraph_fname):
    """Output computed predictions for a specific subgraph to a file. 
    Can be used iteratively in place of predict_subgraphs (e.g., from Jupyter notebook).

    n.b. train_model outputs the info re-computed in predict_subgraphs. This function erases any need to compute a second time.

    Args:
        input_dirpath: data dirpath of a single data split
        split_idx: the chosen fold for the passed labels / predictions
        labels: list of labels for subgraphs
        predictions: list of predictions for subgraphs
        test_dataset: the test TorchOmicsDataset used for the chosen labels / predictioons 
        output_fname: output filename for predictions
    """
    data_split, record = load_data_split_record(input_dirpath)
    
    subgraph_fnames = test_dataset._processed_files
    assert len(labels) == len(predictions) == len(subgraph_fnames)
    subgraph_descriptors_iter = map(_process_subgraph_ids, subgraph_fnames)

    combined_data = [[li, pi, *sgi] for li, pi, sgi in zip(labels, predictions, subgraph_descriptors_iter)] 

    header = ["Label", "Predictions", "Subgraph_ID", "Tissue_ID", "File_Name"]
    ofpath = os.path.join(input_dirpath, output_fname)
    write_header = not os.path.exists(ofpath)
    with open(ofpath, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        if write_header: writer.writerow(header)
        writer.writerows(combined_data)

def load_subgraph_predictions(input_dirpath: str, subgraph_fname: str = _output_combined_subgraph_fname
        ) -> List[Tuple[int, float, str, str, str]]:
    """Load previously computed subgraph predictions saved at a chosen location

    Args:
        input_dirpath: data dirpath of a single data split
        output_fname: output filename for predictions
    
    Returns:
        A list of tuples of form (label, prediction, subgraph ID, tissue ID, file name)
    """

    loaded_data = list()

    subgraphs_fpath = os.path.join(input_dirpath, subgraph_fname)
    with open(subgraphs_fpath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        loaded_data.extend([int(li), float(pi), int(sid), tid, fname] for li, pi, sid, tid, fname in reader)

    return loaded_data

def _process_subgraph_ids(fpaths, sg_pattern: str = "_sg_(\d+)") -> List[Tuple[str, str, str]]:
    """Convert a subgraph id to a tuple containing the tissue ID and filename

    Args:
        fpaths: list of filepaths
        sg_pattern: a regex pattern to split out subgraph

    Returns:
        A list of tuples of form (subgraph ID, tissue ID, file name)
    """
    subgraph_descriptors = list()
    for fpath in fpaths:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        
        subgraph_id_match = re.search(sg_pattern, fname)
        subgraph_id = int(subgraph_id_match.group(1))
        
        sidx = subgraph_id_match.start()
        tissue_id = fname[:sidx]
        
        subgraph_descriptors.append((subgraph_id, tissue_id, fname))

    return subgraph_descriptors 

_strategies = {
            "median": np.median,
            "mean": np.mean,
            "geometric_mean": lambda a: np.power(np.prod(a), 1 / a.size) 
        }
def predict_graphs(input_dirpath: str, combination_strategy: str, 
        subgraphs_fname: str = _output_combined_subgraph_fname, tissues_fname: str = _output_combined_graph_fname
        ) -> List[Tuple[int, float, str]]:
    """Combines subgraph predictions using one of the available combination strategies.
    NOTE: Requires subgraph prediction (predict_subgraphs) prior to predicting graphs

    Args:
        input_dirpath: data dirpath of a single data split
        combination_strategy: method for combining values. Can be `median`, `mean` or `geometric_mean`, corresponding to those functions
        subgraphs_fname: filename for subgraph predictions (previously computed) 
        tissues_fname: filename for graph predictions (output)

    Returns:
        A list of tuples of form (label, prediction, tissue ID)
    """
    if combination_strategy not in _strategies:
        raise ValueError("Subgraph combination strategy not handled.")

    combine_fn = _strategies[combination_strategy]
 
    subgraph_fpath = os.path.join(input_dirpath, subgraphs_fname)
    if not os.path.exists(subgraph_fpath):
        raise ValueError("Subgraph filepath invalid. Run predict_subgraphs (or change the subgraphs_fname argument) to pass pre-predicted subgraphs.")
    
    labels, preds = list(), list()
    tissue_ids = list()

    with open(subgraph_fpath, 'r') as ifile:
        reader = csv.reader(ifile)
        next(reader, None)

        for _lab, _pred, sg_idx, tissue, _ in reader:
            labels.append(int(_lab))
            preds.append(float(_pred))
            tissue_ids.append(tissue)

    sorted_tissue_ids = sorted(set(tissue_ids))
    labels, preds = np.array(labels), np.array(preds)
    tissue_ids = np.array(tissue_ids)

    tissue_labels, tissue_preds = list(), list()
    for ti in sorted_tissue_ids:
        mask = tissue_ids == ti
        tissue_labels.append(labels[mask][0])
        tissue_preds.append(combine_fn(preds[mask]))
    
    combined_data = list(zip(tissue_labels, tissue_preds, sorted_tissue_ids))
    
    print(roc_auc_score(tissue_labels, tissue_preds))
    header = ["Label", "Predictions", "Tissue_ID"]
    ofpath = os.path.join(input_dirpath, tissues_fname)
    with open(ofpath, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(combined_data)

    return combined_data 

def load_graph_predictions(input_dirpath: str, tissue_fname: str = _output_combined_graph_fname) -> List[Tuple[int, float, str]]:
    """Load previously computed graph predictions

    Args:
        input_dirpath: data dirpath of a single data split
        tissues_fname: filename for graph predictions (output)

    Returns:
        A list of tuples of form (label, prediction, tissue ID)
    """
    loaded_data = list()

    tissue_fpath = os.path.join(input_dirpath, tissue_fname)
    with open(tissue_fpath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        loaded_data.extend([int(li), float(pi), tid] for li, pi, tid in reader)

    return loaded_data

def predict_graphs_from_subgraphs(subgraph_ids: List[str], labels: List[int], predictions: List[float], combination_strategy: str = "mean") -> Tuple[list, list, list]:
    """Predicts graphs based on subgraph ID. Helper function.

    Args:
        subgraph_ids: list of subgraph IDs
        labels: subgraph labels corresponding to subgraph_ids list
        predictions: subgraph predictions corresponding to subgraph_ids list
        combination_strategy: method for combining values. Can be `median`, `mean` or `geometric_mean`, corresponding to those functions
    
    Returns:
        Three lists of the same size inccluding graph IDs, graph labels, and graph predictions. 
    """
    combine_fn = _strategies[combination_strategy]

    _per_graph = defaultdict(list)
    for sgi, li, pi in zip(subgraph_ids, labels, predictions):
        gi = sgi.split("sg")[0]
        _per_graph[gi].append((li, pi))

    graph_ids, graph_labels, graph_predictions = list(), list(), list()
    for gi, pred_tuples in _per_graph.items():
        graph_ids.append(gi)
        graph_labels.append(pred_tuples[0][0])
        
        _sg_predictions = np.array([t[1] for t in pred_tuples])
        graph_predictions.append(combine_fn(_sg_predictions))
    
    return graph_ids, graph_labels, graph_predictions
