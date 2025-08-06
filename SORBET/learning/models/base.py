"""An abstract class defining necessary model components for working
with the defined code.
"""
import torch
from abc import abstractmethod

from .model_specification import StructureSpecification 

# This model structure is inspired by: https://github.com/AntixK/PyTorch-VAE

class BaseGraphModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Typical forward method defined in a torch (torch_geometric) function.

        Args:
            x: cell data associated. Equivalent to the Data.x attribute in torch_geomtric.data.Data.
            edge_index: edge indexing. Equivalent to the Data.edge_index attribute in torch_geomtric.data.Data. 
            batch: graph assignment indexing. Equivalent to the Data.batch attribute in torch_geomtric.data.Data.

        Returns: 
            Model output (as logits) for the given graph data. 
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Function to predict outputs for a given input. 
        
        Helpful for predictions of test data or other data where a learning step is not necessary 
        (e.g., gradients not attached).  

        Args:
            x: cell data associated. Equivalent to the Data.x attribute in torch_geomtric.data.Data.
            edge_index: edge indexing. Equivalent to the Data.edge_index attribute in torch_geomtric.data.Data. 
            batch: graph assignment indexing. Equivalent to the Data.batch attribute in torch_geomtric.data.Data.

        Returns: 
            Model predictions (as probabilities) for the given data. 
        """
        raise NotImplementedError

    @abstractmethod
    def get_loss_function(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor: 
        """Computed loss function for the given model. Optimized using the methods in ../train.py.

        Should be called following a step using the forward() method.
        
        Args:
            prediction: predictions for each input graph. Typically, the output of forward(). 
            labels: labels for each graph's prediction. Matches shape of predictions. 
            **kwargs: additional arguments necessary for specific models.
        
        Returns: 
            Loss function computed for a given set of graphs and predictions.  
        """
        raise NotImplementedError

    @abstractmethod
    def get_cell_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Function to extract the cell embeddings of a series of cells. 
        
        This is usually the last representation before a pooling step. 

        Args:
            x: cell data associated. Equivalent to the Data.x attribute in torch_geomtric.data.Data.
            edge_index: edge indexing. Equivalent to the Data.edge_index attribute in torch_geomtric.data.Data. 
            batch: graph assignment indexing. Equivalent to the Data.batch attribute in torch_geomtric.data.Data.

        Returns: 
            Computed cell embedding of input cells. Of same row size as x. 
        """
        raise NotImplementedError

    @abstractmethod
    def get_subgraph_embedding(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Function to extract the subgraph embeddings of a series of input graphs.
        
        This is usually a representation after the last pooling step.
        
        Args:
            x: cell data associated. Equivalent to the Data.x attribute in torch_geomtric.data.Data.
            edge_index: edge indexing. Equivalent to the Data.edge_index attribute in torch_geomtric.data.Data. 
            batch: graph assignment indexing. Equivalent to the Data.batch attribute in torch_geomtric.data.Data.

        Returns: 
            Model predictions for the given data. 
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_model_specification() -> StructureSpecification:
        """A static method for each class returning the class's specification. 

        Necessary for hyperparameter opt. See also model_specification.py for a description of how specifications are defined.
        
        Returns:
            A StructureSpecification object describing the models input parameters.
        """
        raise NotImplementedError
