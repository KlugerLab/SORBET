import os
import csv, pickle
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class ExperimentRecord:
    """A data class detailing a file structure to save values to.
    """
    data_split: List[List[str]]
    data_split_file: str
    output_dir: str
    tensorboard_dir: str
    ray_dir: str
    baselines_dir: str
    models_dir: str
    perf_dir: str
    plots_dir: str


_split_fname = "data_split.p"
_tensorboard_dname = "tensorboard"
_ray_dname = "ray_tuning"
_baselines_dname = "baselines"
_models_dname = "models"
_perf_dname = "perf"
_plots_dname = "plots"

def create_data_split_record(split: List[Tuple[list]], output_directory: str) -> Tuple[List[Tuple[list]], ExperimentRecord]:
    """Creates a new directory with a pre-defined structure for saving SORBET experiments on a specific data split.

    Args:
        split: data split for the chosen analysis
        output_directory: directory path to output data split

    Returns:
        A tuple of two values, the data split and the associated ExperimentRecord.
        
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    record = _get_directory_structure(output_directory)

    for dirpath in [record.tensorboard_dir, record.ray_dir, record.baselines_dir, record.models_dir, record.perf_dir, record.plots_dir]:
        os.makedirs(dirpath)
    
    record.data_split = split
    with open(record.data_split_file, 'wb+') as ofile:
        pickle.dump(split, ofile)

    return split, record 

# A bit confused how this is called after create_data_split_record if create_data_split_record Creates a new directory with a pre-defined structure. probably the latter docstring needs an update.
def load_data_split_record(directory: str):
    """Loads a previously dumped data split from a pre-specified directory structure. 
    NOTE: Called after create_data_split_record.

    Args:
        directory: the filepath where the previously saved record was created
 
    Returns:
        A tuple of two values, the data split and the associated ExperimentRecord.
    """

    record = _get_directory_structure(directory)
    
    split = pickle.load(open(record.data_split_file, 'rb'))
    record.data_split = split

    return split, record

def _get_directory_structure(dirpath: str):
    """Structures an ExperimentRecord object with the given directory structure relative to the input dirpath.

    Args:
        dirpath: the directory path where experiments should be saved.

    Returns:
        The created ExperimentRecord value.
    """
    dirpath = os.path.abspath(dirpath)
    assert os.path.exists(dirpath)

    tensorboard_dirpath = os.path.join(dirpath, _tensorboard_dname)
    ray_dirpath = os.path.join(dirpath, _ray_dname)
    baselines_dirpath = os.path.join(dirpath, _baselines_dname)
    models_dirpath = os.path.join(dirpath, _models_dname)
    perf_dirpath = os.path.join(dirpath, _perf_dname)
    plots_dirpath = os.path.join(dirpath, _plots_dname)

    data_split_fpath = os.path.join(dirpath, _split_fname)

    return ExperimentRecord(
            data_split = None,
            data_split_file = data_split_fpath,
            output_dir = dirpath,
            tensorboard_dir = tensorboard_dirpath,
            ray_dir = ray_dirpath,
            baselines_dir = baselines_dirpath,
            models_dir = models_dirpath,
            perf_dir = perf_dirpath,
            plots_dir = plots_dirpath
            )
