"""Dataclass for hyperparameter specification, per model. 
"""
from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass 
class StructureSpecification:
    """Dataclass used to specify what parameters are optimizable for a given model.
    Necessary for the hyperparameter optimization functions used in training.

    See specific models for examples of implementation.
    
    Serializable parameters are those which may come in series of parameters. These parameters
        all define layers in a network and, specifically, the depth and width of a specific part
        of the network. 

    Non-serializable parameters are single parameters, which may be set by a single sample. An 
        example would be a learning rate or a dropout value. Additional information must typically
        be passed to define the type of samples that might be drawn (e.g., float from a log-range).
    
    Please see their definitions in model classes (e.g., sorbet_gcn.py) and their parsing in 
        ../hyperparameter_optimization.py:_define_by_run_func and 
        ../hyperparameter_optimization_utils.py:_parse_hyperparameter_type

    Attributes:
        model_serializable_parameters: a list of lists of serializable parameters.
        model_non_serializable_parameters: a list of lists of non serializable parameters
        model_input_specifier: string to specify which argument sets the input size
        random_seed_specified: string to specify which model argument sets a random seed

    """
    model_serializable_parameters: List[List[Union[bool, str, int]]]
    model_non_serializable_parameters: List[List[Union[bool, str, int]]]
    model_input_specifier: str = "in_channel"
    random_seed_specifier: str = "random_seed" 
