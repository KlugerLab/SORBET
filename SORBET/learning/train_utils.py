"""Helper functions for train.py
"""
from typing import List, Dict, Tuple, Any, Iterable
import csv, os, pickle
import numpy as np
import torch

def stratified_kfold_split(labels_or_fpath: Any, k: int, include_validation = False, as_fpath = False) -> Iterable[List[list]]:
    """Computes stratified K-fold CV split (as generator)

    Args:
        labels_or_fpath: either a dictionary or str (filepath)
            If a dictionary, a map between graphs and labels
            If a string, its the filepath recording files and labels
        k: number of folds to split the data into
        include_validation: whether to include a validation set in the data
        as_fpath: whether labels_or_fpath was passed as a filepath

    Returns:
        Iterator of tuples of two or three lists. 
            The first list is the train set, the last set the test set and, if included, the middle set is the validation set.
    """
    if as_fpath:
        positives, negatives = _load_samples_split(labels_or_fpath)
    else:
        positives = [k for k,v in labels_or_fpath.items() if v != 0]
        negatives = [k for k,v in labels_or_fpath.items() if v == 0]

    positives = np.array_split(np.random.permutation(positives), k)
    negatives = np.array_split(np.random.permutation(negatives), k)
    
    make_str = lambda t: list(map(str, t))

    for ki in range(k):
        test_set = [*positives[ki], *negatives[ki]]

        if include_validation:
            val_idx = (ki + 1) % k
            val_set = [*positives[val_idx], *negatives[val_idx]]

            train_set = list()
            for kj in filter(lambda kj: kj != ki and kj != val_idx, range(k)):
                train_set.extend([*positives[kj], *negatives[kj]])
            
            yield make_str(train_set), make_str(val_set), make_str(test_set)
        else:
            train_set = list()
            for kj in filter(lambda kj: kj != ki, range(k)):
                train_set.extend([*positives[kj], *negatives[kj]])
            
            yield make_str(train_set), make_str(test_set)

def repeated_stratified_kfold_splits(labels_or_fpath: Any, k: int, r: int, include_validation = False, as_fpath = False) -> Iterable[List[list]]:
    """Computes repeated leave a fold out CV (as generator)
    
    Args:
        labels_or_fpath: either a dictionary or str (filepath)
            If a dictionary, a map between graphs and labels
            If a string, its the filepath recording files and labels
        k: number of folds to split the data into
        r: number of times to repeat the data spllit
        include_validation: whether to include a validation set in the data
        as_fpath: whether labels_or_fpath was passed as a filepath

    Returns:
        Iterator of tuples of two or three lists. 
            The first list is the train set, the last set the test set and, if included, the middle set is the validation set.
    """
    if as_fpath:
        positives, negatives = _load_samples_split(labels_or_fpath)
    else:
        positives = [k for k,v in labels_or_fpath.items() if v != 0]
        negatives = [k for k,v in labels_or_fpath.items() if v == 0]

    pos_fold_size = len(positives) // k
    neg_fold_size = len(negatives) // k

    for ni in range(r):
        test_set = [*np.random.choice(positives, pos_fold_size), *np.random.choice(negatives, neg_fold_size)]
        
        if include_validation:
            _positives = [pi for pi in positives if pi not in test_set]
            _negatives = [ni for ni in negatives if ni not in test_set]
            val_set = [*np.random.choice(_positives, pos_fold_size), *np.random.choice(_negatives, neg_fold_size)]
            
            _positives = [pi for pi in _positives if pi not in val_set]
            _negatives = [ni for ni in _negatives if ni not in val_set]
            
            yield [*_positives, *_negatives], val_set, test_set
        else:
            _positives = [pi for pi in positives if pi not in test_set]
            _negatives = [ni for ni in negatives if ni not in test_set]

            yield [*_positives, *_negatives], test_set

def make_inner_kfold_split(data_split: List[Tuple[list]], excluded_index: int) -> Tuple[List[str], List[str]]:
    """Helper function to re-split a k-fold split to nest a k-fold split 
    
    This function is helpful for nested cross validation. For example, 
    Consider a five fold cross validation with test splits, {0,1,2,3,4}, where each index corresponds to a set of graph IDs.

    stratified_kfold_split creates a record like: 
        [[[0,1,2], [3], [4]], [[4,0,1],[2],[3]],...]
    make_inner_kfold_split with, for example, excluded_index = 4, creates a data split like:
        [[[0,1,2],[3],[4]], [[0,1,3],[2],[4]], [[0,2,3],[1],[4]], [[1,2,3],[0],[4]]] 
    (All possible sub-splits of the (k-1) folds not being tested).    

    Args:
        data_split: the outer k-fold split to re-split along inner splits
        excluded_index: the test fold to exclude from the inner re-split 

    Returns:
        Two lists, an innner k-fold split, and the training folds used to generate those folds
    """
    train_folds = [ds[-1] for i, ds in enumerate(data_split) if i != excluded_index]
    test_fold = data_split[excluded_index][-1]

    inner_data_split = list()
    for i in range(len(data_split) - 1):
        val_fold = train_folds[i]
        train_fold = [sample_id for j, samples in enumerate(train_folds) for sample_id in samples if j != i]

        inner_data_split.append((train_fold, val_fold, test_fold))
        assert all(ti not in test_fold for ti in train_fold)
        assert all(vi not in test_fold for vi in val_fold)

    return inner_data_split, train_folds

def make_inner_loo_split(labels_or_fpath: Any, test_labels: List[str], sub_sample: int = None, as_fpath = False) -> List[Tuple[list]]:
    """Makes an inner leave-one-out split. Can sub-sample to `sub_sample` total examples. 

    Args:
        labels_or_fpath: either a dictionary or str (filepath)
            If a dictionary, a map between graphs and labels
            If a string, its the filepath recording files and labels
        test_labels: graph keys left out
        sub_sample: number of LOO splits left in 
        as_fpath: whether labels_or_fpath was passed as a filepath
    
    Returns:
        Iterator of tuples of two or three lists. 
            The first list is the train set, the last set the test set and, if included, the middle set is the validation set.
    """
    if as_fpath:
        positives, negatives = _load_samples_split(labels_or_fpath)
    else:
        positives = [k for k,v in labels_or_fpath.items() if v != 0]
        negatives = [k for k,v in labels_or_fpath.items() if v == 0]

    positives = [gi for gi in positives if gi not in test_labels]
    negatives = [gi for gi in negatives if gi not in test_labels]
    
    n_splits = min(len(positives), len(negatives))
    
    _pos_split = np.array_split(np.random.permutation(positives), n_splits)
    _neg_split = np.array_split(np.random.permutation(negatives), n_splits)
    
    loo_split = list()
    for i in range(n_splits):
        _train = list()
        for j in range(n_splits):
            if i == j: continue

            _train.extend(map(str, _pos_split[j]))
            _train.extend(map(str, _neg_split[j]))

        _val = list(map(str, [*_pos_split[i], *_neg_split[i]]))
        
        loo_split.append([_train, _val, test_labels])

    if sub_sample is not None:
        sample = np.random.choice(len(loo_split), sub_sample, replace=False)
        loo_split = [loo_split[idx] for idx in sample]

    return loo_split


def _load_samples_split(labels_fpath) -> Tuple[list, list]:
    """Returns two lists of sample ids split by positive and negative sample
    
    Args:
        labels_fpath: filepath of the labels file

    Returns:
        Two lists of the positively and negatively labeled graphs
    """
    with open(labels_fpath, 'r') as ifile:
        reader = csv.reader(ifile, delimiter=',')
        next(reader)
        
        positives, negatives = list(), list()
        for row in reader:
            if int(row[1]) != 0:
                positives.append(row[0])
            else:
                negatives.append(row[0])
    
    return positives, negatives

# NOTE: Not declared at the top of file to discourage changing the filepaths. These files are appended to an input function
# (e.g., fpath_rt) to create two files (e.g., fpath_rt + _model_ext is the saved model state; fpath_rt + _kwargs_ext the initialization argument)
_model_ext = '_model_statedict.pt'
_kwargs_ext = '_model_init.p'
def save_model(model, kwargs: Dict[str, Any], fpath_rt: str, model_ext: str = _model_ext, kwargs_ext: str = _kwargs_ext):
    """Saves a model to a defined location. Requires multiple files saved.
    Re-load model using the load_model function.

    Args:
        model: model to be saved
        kwargs: keyword arguments used to intialize the model
        fpath_rt: filepath root to save files to
        model_ext: file extension for the model's core file. Default: _model_statedict.p
        kwargs_ext: file extension for the model's keyword argument file. Default: _model_init.p
    """
    model_fpath = fpath_rt + model_ext 
    torch.save(model.state_dict(), model_fpath)
    
    kwargs_fpath = fpath_rt + kwargs_ext 
    with open(kwargs_fpath, 'wb+') as f:
        pickle.dump(kwargs, f)

def load_model(model_type, fpath_rt: str, model_ext: str = _model_ext, kwargs_ext: str = _kwargs_ext):
    """Loads a model saved to a defined location. 
    Re-load model saved using the save_model function.

    Args:
        model_type: model type of the saved model 
        fpath_rt: filepath root to save files to
        model_ext: file extension for the model's core file. Default: _model_statedict.p
        kwargs_ext: file extension for the model's keyword argument file. Default: _model_init.p

    Returns:
        A model saved at the fpath_rt location.
    """

    model_fpath = fpath_rt + model_ext 
    kwargs_fpath = fpath_rt + kwargs_ext

    if not os.path.exists(model_fpath) or not os.path.exists(kwargs_fpath):
        raise ValueError("Could not find model or kwargs file in given location.")
    
    with open(kwargs_fpath, 'rb') as f:
        kwargs = pickle.load(f)
    
    model = model_type(**kwargs)
    
    state_dict = torch.load(model_fpath)
    model.load_state_dict(state_dict)
    
    model.eval()

    return model
