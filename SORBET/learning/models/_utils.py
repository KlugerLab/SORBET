"""Contains general utilities use by all model classes.
"""
from torch_geometric.nn import pool


_pooling_functions = {
        "global_mean": pool.global_mean_pool,
        "global_max": pool.global_max_pool,
        "global_add": pool.global_add_pool
        }

def parse_pooling_function(pooling_function: str):
    """Maps pooling function arguments to the associated pooling_function. 
    
    See the dictionary, _pooling_functions, defined above. Mapping includes functions
        pool.global_mean_pool, pool.global_max_pool, pool.global_add_pool

    Args:
        pooling_function: a string corresponding to the associated pooling function in torch_geometric.nn.pool

    Returns:
        The pooling function corresponding to the input argument. 
    """
    # NOTE: Personal preference would be to pass functions directly to the class constructor. This does not play well with ray / optuna.
    assert pooling_function in _pooling_functions
    return _pooling_functions[pooling_function]
