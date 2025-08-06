"""Data object for working with CCA datasets. Requires two datasets with matching second dimensions. 
"""
from typing import List
import numpy as np
from torch.utils.data import Dataset

class CCAData(Dataset):
    """PyTorch Dataset object for handling CCA data. Includes attributes:
    - X1: The first dataset. Has size (d1 x n)
    - X2: The second dataset. Has size (d2 x n)
    - N: The second dimension size. Here, typically defined number of cells.
    """
    def __init__(self, X1: np.ndarray, X2: np.ndarray):
        """
        Args:
            X1, X2: input datasets for downstream CCA.
        """
        self.X1 = X1
        self.X2 = X2

        self.N = X1.shape[1]

    def __len__(self) -> int:
        """
        Returns:
            The shared dimension's size for X1 and X2.
        """
        return self.N

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        """
        Args:
            idx: column index to return, corresponding to the embeddings of a single cell.

        Returns:
            Two arrays, each of length N, corresponding to the chosen index 
        """
        return self.X1[:,idx], self.X2[:,idx]
