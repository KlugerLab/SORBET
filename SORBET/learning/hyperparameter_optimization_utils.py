"""Helper functions for hyperparameter_optimization.py
"""
from typing import List, Dict, Any, Union, Optional
from functools import partial

from optuna.trial import Trial

from .train import _training_hyperparamters_types 

def get_model_and_training_specifications(config: Dict[str, any], model_type: Any, in_data_dimension: int, random_seed: Optional[int] = None
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Converts a ray / optuna config directory into a structure accessible for training

    Args:
        config: dictionary of configuration choices. See models/ specifications.
        model_type: the choice of model to train. See models/.
        in_data_dimension: size of input data
        random_seed: set a random seed 

    Returns:
        Two dictionaries encoding hyperparameters and model_structure. 
            hyperparameters are typically the remaining keyword arguments in train_model in train.py
            model_structure is typically passed to train_model as model_init_params 
    """
    params = dict()
    for hparam_type in filter(lambda t: t in config, map(lambda t: t[0], _training_hyperparamters_types)):
        params[hparam_type] = config[hparam_type]
    
    model_specification = model_type.get_model_specification()
    
    model_structure = dict()
    model_structure[model_specification.model_input_specifier] = in_data_dimension
    model_structure[model_specification.random_seed_specifier] = random_seed

    for param_key in model_specification.model_serializable_parameters:
        model_structure[param_key] = _deserialize_layer_structure(param_key, config)

    for param_key, _, _ in filter(lambda k: k[0] in config, model_specification.model_non_serializable_parameters):
        model_structure[param_key] = config[param_key]
    
    return params, model_structure

# NOT CRITICAL: what happens when you want to test two values? (in terms of space). I assume you can set in the type but not clear from doc
def _parse_hyperparameter_type(trial: Trial, space: List[Any], hparam_name: str, hparam_type: Any, hparam_log_space: bool): 
    """Parses model specification and suggests the appropriate values for a given trial.
    
    If the size of the search space, <space>, is two, assumes that the values represent bounds on the range.
    Otherwise, treats <space> as a categorical list to suggest from.

    Args:
        trial: an Optuna Trial object parameters are sugggested to
        space: the parameter space to sample from
        hparam_name: name of the sampled hyperparameter
        hparam_log_space: whether the value is sampled from a logarithmic space
    """
    # Implements assumption of categorical choice or range depending on the length of <space>
    _choice_or_range = lambda f: f(hparam_name, *space) if len(space) == 2 else trial.suggest_categorical(hparam_name, space)

    if hparam_type == int:
        _choice_or_range(trial.suggest_int)
    elif hparam_type == float:
        _choice_or_range(partial(trial.suggest_float, log = hparam_log_space))
    elif hparam_type == 'choice':
        trial.suggest_categorical(hparam_name, space)
    else:
        raise ValueError(f'Could not parse specification for {hparam_name}')


def _serialize_layer_structure(trial: Trial, search_param: str, n_layers: int, param_space: List[int], categorical: bool): 
    """Converts a search space over variable layer depth and width to a config for use with Optuna 

    Args:
        trial: an Optuna Trial object parameters are sugggested to
        search_param: name of sampled layer
        n_layers: number of layers to sample structure from
        param_space: params to sample from. Either endpoints of a range or, if categorical, a list to sample from.
        categorical: whether the value is sampled from a categorical space
    """
    for idx in range(n_layers):
        if categorical:
            trial.suggest_categorical(f'{search_param}_{idx}', param_space)
        else:
            trial.suggest_int(f'{search_param}_{idx}', *param_space) 

def _deserialize_layer_structure(search_param: str, config: Dict[str, Union[float, int]]):
    """Converts serialized data from _serialize_layer_structure into a (correctly-ordered) list

    Args:
        search_param: name of sampled layer to extract sampled size frmo
        config: sampled configuration
    """
    unordered_param_list = list()
     
    for k, v in filter(lambda t: search_param in t[0], config.items()):
        if 'n_layers' in k: continue
        idx = int(k.split("_")[-1])
        unordered_param_list.append((idx, v))
    
    sorted_layers = sorted(unordered_param_list, key=lambda x: x[0])
    param_list = [val for _, val in sorted_layers]
    
    return param_list
