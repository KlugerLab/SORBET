"""Dataset objects for use with torch / torch_geometric in training models.
"""
import os
import csv
from dataclasses import dataclass
from typing import List, Union

import torch
from torch_geometric.data import Data, Dataset

class TorchOmicsDataset(Dataset):
    """The base torch_geometric dataset use for feeding data in to model training procedures

    Attributes:
        _processed_dir: Directory path for loaction of processed files.
        _processed_files: List of processed files. Used in conjunction w/ _processed_dir.

    """
    def __init__(self, root: str, subgraph_metadata: SubgraphMetadata, split: List[str] = None, transform=None, pre_transform=None, pre_filter=None):
        """Initializes a TorchOmicsDataset.

        Largely follows the standard format (similar to tutorials) prescribed by the torch_geometric package.

        Args:
            root: root directory. All filepaths are relative to this path.
            subgraph_metadata: SubgraphMetadata object defining the location of the processed data. Relates to the structure
                formed in data_handling/preprocess.py
            split: list of graph ids. Used to select a subset of files to include (e.g., for cross validation) 
            transform: torch_geomtric data standard. Not used.
            pre_transform: torch_geomtric data standard. Not used.
            pre_filter: torch_geomtric data standard. Not used.
        """
        super().__init__(root, transform, pre_transform, pre_filter)

        self._processed_dir = subgraph_metadata.processed_dirpath
        self._processed_files = _load_filtered_graphs(split, root, subgraph_metadata.subgraph_map, subgraph_metadata.torch_subgraph_map)

    @property
    def processed_dir(self) -> str:
        """Getter function for the location of processed torch graphs

        Returns:
            Processed file directory path.
        """
        return self._processed_dir 
    
    @property
    def processed_file_names(self) -> List[str]:
        """Getter function for the processed torch graph files.

        Returns:
            Processed torch file names.
        """
        return self._processed_files

    def len(self) -> int:
        """Function computing the number of subgraphs included in this dataset.

        Note, this can vary depending on the specific graphs identified using the `split` initialization argument.

        Returns:
            Number of graphs included in this dataset. 
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """Get a specific subgraph based on indexing in the _processed_files list.

        Args:
            idx: index in the list of processed files returned by processed_file_names
        
        Returns:
            A torch_geomtric Data object representing a specific subgraph.    
        """

        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data

@dataclass
class SubgraphMetadata:
    """Dataclass defining paths to subgraphs (both OmicsGraph and Data).
    
    Please see data_handling/preprocess.py for a description of directory structure. 

    Attributes:
        processed_dirpath: the (relative) path to the specific set of subgraphs
        subgraph_map: a filename in processed_dirpath mapping graphs to their related subgraphs. All data_handling.OmicsGraph objects
        torch_subgraph_map: a filename in processed_dirpath mapping data_handling.OmicsGraph objects to torch_geometric.data.Data objects.
    """
    processed_dirpath: str
    subgraph_map: str
    torch_subgraph_map: str

def make_subgraph_metadata(processed_dirpath: str, py_mapping_fname: str = "py_subgraph_mapping.csv", 
        torch_mapping_fname: str = "torch_subgraph_mapping.csv") -> SubgraphMetadata:
    """Helper function to make the SubgraphMetadata class. 
    
    The py_mapping_fname and torch_mapping_fname should not need to be changed (unless modified during pre-processing).
    
    Args:
        processed_dirpath: directory path of pre-processed data
        py_mapping_fname: csv file mapping full OmicsGraph objects to associated OmicsGraph subgraph objects. 
        torch_mapping_fname: csv file mapping OmicsGraph to Data objects. 

    Returns: 
        A SubgraphMetadata object encoding the position of processed data, as defined in SubgraphMetadata.
    """
    py_map_fpath = os.path.join(processed_dirpath, py_mapping_fname)
    torch_map_fpath = os.path.join(processed_dirpath, torch_mapping_fname)

    assert os.path.exists(py_map_fpath)
    assert os.path.exists(torch_map_fpath)
    
    return SubgraphMetadata(os.path.abspath(processed_dirpath), py_map_fpath, torch_map_fpath)

def _load_filtered_graphs(included_graphs: Union[List[str], None], datadir: str, subgraph_mapping_fpath: str, torchgraph_mapping_fpath: str) -> List[str]:
    """Returns a list of filepaths to torch files representing subgraphs. 
    
    Args:
        included_graphs: subset of subgraphs included for a specific dataset. 
            If list is of length 0 or is None, all possible subgraphs are included.
        datadir: directory where pre-processed subgraphs are stored
        subgraph_mapping_fpath: filename of OmicsGraph subgraphs mapping
        torchgraph_mapping_fpath: filename of torch.data.Data subgraphs mapping

    Returns:
        A list of file paths to torch_geomtric.data.Data for the selected subgraphs.
    """
    if included_graphs is None or len(included_graphs) == 0:
        with open(os.path.join(datadir, torchgraph_mapping_fpath), 'r') as ifile:
            reader = csv.reader(ifile, delimiter=',')
            next(reader)

            torchgraphs = [t[1] for t in reader]
    else:
        subgraphs = list()
        with open(os.path.join(datadir, subgraph_mapping_fpath), 'r') as ifile:
            reader = csv.reader(ifile, delimiter=',')
            next(reader)

            for row in filter(lambda t: t[0] in included_graphs, reader):
                subgraphs.extend(row[2:])
        
        with open(os.path.join(datadir, torchgraph_mapping_fpath), 'r') as ifile:
            reader = csv.reader(ifile, delimiter=',')
            next(reader)

            torchgraphs = [t[1] for t in reader if t[0] in subgraphs]
    
    return torchgraphs
