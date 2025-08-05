import os
import csv, pickle
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class LearningRecord:
    """Data structure for saving reasoning outputs 
    """
    model_dir: str
    statistics_dir: str
    embeddings_dir: str
    plots_dir: str
    cell_embedding_file: str
    subgraph_embedding_file: str

_model_dname = "models"
_statistics_dname = "statistics"
_embeddings_dname = "embeddings"
_plots_dname = "plots"
_cell_embeddings_fname = "cell_embeddings.npz"
_sg_embeddings_fname = "sg_embeddings.npz"

# might be useful to add here a description of the structure and what the use might find in each sub directory or if these are all intermidate/temp files mention it explicitly
def create_data_split_record(output_directory: str) -> LearningRecord:
    """Creates a new directory with a pre-defined structure for saving SORBET experiments on a specific data split.

    Args:
        output_directory: a directory path to output data to. Directory is created if it does not exist.

    Returns:
        A LearningRecord object encoding the structure at output_directory 
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    record = _get_directory_structure(output_directory)

    for dirpath in [record.model_dir, record.statistics_dir, record.embeddings_dir, record.plots_dir]:
        if not os.path.exists(dirpath): os.makedirs(dirpath)
   
    return record 

def load_data_split_record(input_directory: str) -> LearningRecord:
    """Loads a previously dumped data split from a pre-specified directory structure. 
    Called after a record is instantiated using create_data_split_record

    Args:
        input_directory: a directory path to output data to. 

    Returns:
        A LearningRecord object encoding the structure at input_directory 
    """
    record = _get_directory_structure(input_directory)
    return record

def _get_directory_structure(dirpath: str) -> LearningRecord:
    """Structures an ExperimentRecord object with the given directory structure relative to the input dirpath.

    Args:
        dirpath: a directory path to output data to.

    Returns:
        A LearningRecord object encoding the structure at input_directory 
    """
    dirpath = os.path.abspath(dirpath)
    assert os.path.exists(dirpath)
    
    model_dirpath = os.path.join(dirpath, _model_dname)
    statistics_dirpath = os.path.join(dirpath, _statistics_dname)
    embeddings_dirpath = os.path.join(dirpath, _embeddings_dname)
    plots_dirpath = os.path.join(dirpath, _plots_dname)
    cell_embedding_fpath = os.path.join(dirpath, _cell_embeddings_fname)
    sg_embedding_fpath = os.path.join(dirpath, _sg_embeddings_fname)

    return LearningRecord(
            model_dir = model_dirpath,
            statistics_dir = statistics_dirpath,
            embeddings_dir = embeddings_dirpath,
            plots_dir = plots_dirpath,
            cell_embedding_file = cell_embedding_fpath,
            subgraph_embedding_file = sg_embedding_fpath
            )
