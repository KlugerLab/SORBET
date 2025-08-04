import os
from typing import List, Dict, Union, Any, Optional
from dataclasses import dataclass
from functools import partial
from tqdm import tqdm as tqdm

import numpy as np
from sklearn.metrics import roc_auc_score

import optuna
import ray
from ray import tune
from ray.air import ScalingConfig, RunConfig
from ray.tune.search.optuna import OptunaSearch

from optuna.samplers import RandomSampler, TPESampler, CmaEsSampler 

from .dataset import TorchOmicsDataset
from .train import train_model, _training_hyperparamters_types 
from .hyperparameter_optimization_utils import _serialize_layer_structure, _deserialize_layer_structure, _parse_hyperparameter_type
from .hyperparameter_optimization_utils import get_model_and_training_specifications 

from .predict import predict_graphs_from_subgraphs

def objective(config: Dict[str, any], data_split: List[Tuple[list]], root_fpath: str, metadata_files: Any, 
        model_type: Any, in_data_dimension: int, random_seed: Optional[int] = None, tensorboard: Optional[str] = None):
    """The standard objective function, defined for use with Ray / Optuna.
    Estimated performance is reported using the tune framework (not returned), as is typical.

    Args:
        config: the config to evaluate. Passed via ray tune framework
        data_split: list of data splits. See train_utils.py for typical outputs
        root_fpath: the root of the data directories
        metadata_files: a SubgraphMetadata data class defining the data structure in root_fpath 
        model_type: model class that optimization is applied to. See models/ 
        in_data_dimension: input data dimension to models
        random_seed: optional capacity to set random seed
        tensorboard: tensorboard filepath to save output values to
    """
    train_params, model_structure = get_model_and_training_specifications(config, model_type, in_data_dimension, random_seed) 
    
    graph_aurocs = list()
    aurocs = list()
    # The third argument is the test set -- ignored.
    for train_ids, val_ids, _ in data_split:
        train_ds = TorchOmicsDataset(root_fpath, metadata_files, train_ids)
        test_ds = TorchOmicsDataset(root_fpath, metadata_files, val_ids)
    
        train_params['validate_step'] = train_params['epochs'] - 2 # Do not need to evaludate validation step.
        _, (_preds, _labs), acc, tl, vl = train_model(model_type, model_structure, [train_ds, test_ds], **train_params, 
                print_lvl=0, tensorboard_dir=tensorboard, random_seed = random_seed)
    
        _auroc = roc_auc_score(_labs, _preds)
        aurocs.append(_auroc)
    
        _, _ls, _ps = predict_graphs_from_subgraphs(test_ds._processed_files, _labs, _preds)
        _auroc_comb = roc_auc_score(_ls, _ps)
        graph_aurocs.append(_auroc_comb)

    mean_auroc = np.mean(aurocs)
    tune.report(auroc=mean_auroc) 


# we can add a warning if this is violated: "n. GPUs <= n. CPUs"
def hyperparameter_optimization(data_split: List[Tuple[list]], root_fpath: str, metadata_files: Any, input_data_dimension: int,  
        model_type: Any, model_hyperparameters: Dict[str, Any], set_model: bool = False, 
        tensorboard_dir: str = None, ray_run_config_kwargs: dict = {'verbose': 1},
        num_model_evals: int = 100, n_cpus: int = 12, n_gpus: int = 6, allow_fractional: float = 0, random_seed: Optional[int] = None) -> tune.ResultGrid:
    """The central function for hyperparameter optimization. Called to set-up and run Ray-tuning.

    Args:
        data_split: list of data splits. See train_utils.py for typical outputs
        root_fpath: the root of the data directories
        metadata_files: a SubgraphMetadata data class defining the data structure in root_fpath 
        input_data_dimension: input data dimension to models
        model_type: model class that optimization is applied to. See models/ 
        model_hyperparameters: dictionary of model hyperparameters used for defining the optimization framework.
        set_model: whether the structure of the model (n. layers, width of layers, etc) is pre-set. If not, ray optimizes over passed hyperparameters.
        tensorboard_dir: tensorboard filepath to save output values to
        ray_run_config_kwargs: additional keyword arguments passed to Ray.
        num_model_evals: number of sampled trials to run.
        n_cpus: number of CPUs used during optimzation
        n_gpus: number of GPUs used during evaluation. Assumes n_cpus >= n_gpus.
        allow_fractional: allow multiple models on each GPU.
        random_seed: a random seed to set for training all models

    Returns:
        A ray tune ResultGrid object summarizing the results of the experiment. 
    """

    # Identify available resources and infer concurrent jobs:
    # NOTE: Assumes that n. GPUs <= n. CPUs. This seems like the right assumption generally, but may need to be addressed in future iterations. 
    if allow_fractional != 0 and allow_fractional < 1:
        n_concurrent = int(n_gpus // allow_fractional)
        per_trial = {'gpu': n_gpus / n_concurrent, 'cpu': n_cpus / n_concurrent}
    else:
        n_concurrent = min(n_gpus, n_cpus)
        per_trial = {'gpu': n_gpus // n_concurrent, 'cpu': n_cpus // n_concurrent}
    
    # Initialize Ray instance:
    ray.init()

    # Set-up hyperparamter search space functions:
    model_specification = model_type.get_model_specification()
    hparam_suggestion_fn = partial(_define_by_run_func, model_hyperparameters=model_hyperparameters, model_specification=model_specification, pre_defined_model = set_model) 
    search = OptunaSearch(
            space = hparam_suggestion_fn,
            metric = "auroc",
            mode = "max",
    )

    # Run tuning.
    tuner = tune.Tuner(
            trainable = tune.with_resources(
                tune.with_parameters(
                    objective,
                    data_split = data_split,
                    root_fpath = root_fpath,
                    metadata_files = metadata_files,
                    model_type = model_type,
                    in_data_dimension = input_data_dimension,
                    random_seed = random_seed,
                    tensorboard = tensorboard_dir
                    ),
                resources = per_trial, 
            ),
            tune_config = tune.TuneConfig(
                search_alg = search,
                num_samples = num_model_evals,
                max_concurrent_trials = n_concurrent 
            ),
            run_config = RunConfig(**ray_run_config_kwargs)
    )

    results = tuner.fit()
    
    # Shutdown Ray instance:
    ray.shutdown()

    return results

def _define_by_run_func(trial: optuna.trial, 
        model_hyperparameters: Dict[str, Union[Any, List[Any]]], model_specification: Dict[str, Union[Any, List[Any]]], 
        pre_defined_model: bool = False) -> Optional[Dict[str, Any]]:
    """A function for defining a new model trial. 
    Following example defined in: https://docs.ray.io/en/latest/tune/examples/optuna_example.html
    
    Args:
        trial: an Optuna trial object
        model_hyperparameters: a dictionary mapping hyperparameter labels (as defined in the model's specification) to sampling choices. 
        model_specification: the chosen models specification. See models/model_specification.py
        pre_defined_model: whether the structure of the model (n. layers, width of layers, etc) is pre-set. If not, ray optimizes over passed hyperparameters.

    Returns:
        Optionally returns a dictionary of constants in the configuration. Constants are typically the pre-defined model structure and hyperparameters that 
            are not optimized.
    """
    config_constants = dict()
   
    # Parse model layers (potentially variable number of layers):
    for sparam in model_specification.model_serializable_parameters:
        if pre_defined_model:
            layers = model_hyperparameters[sparam]
            config_constants[f'{sparam}_n_layers'] = len(layers)
            for idx, size in enumerate(layers):
                config_constants[f'{sparam}_{idx}'] = size
        else:
            _spec = model_hyperparameters[sparam]
            if len(_spec) == 3:
                n_low, n_high, _pspace = _spec
                cat = True
            else:
                n_low, n_high, min_size, max_size = _spec
                _pspace = [min_size, max_size]
                cat = False
                
            n_layers = trial.suggest_int(f'{sparam}_n_layers', n_low, n_high) 
            _serialize_layer_structure(trial, sparam, n_layers, _pspace, cat)  
    
    # Defines over training hyperparameters (_training_hyperparamters_types) and model specification hyperparameters.
    parameter_definitions = [*_training_hyperparamters_types, *model_specification.model_non_serializable_parameters] 
    for param_name, param_type, param_log_space in filter(lambda t: t[0] in model_hyperparameters, parameter_definitions):
        param_space = model_hyperparameters[param_name]
        if len(param_space) == 1: # Set as constant if length of space is 1
            config_constants[param_name] = param_space[0]
        else:
            _parse_hyperparameter_type(trial, param_space, param_name, param_type, param_log_space)

    if len(config_constants) > 0:
        return config_constants
