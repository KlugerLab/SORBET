"""This file implements a helper method for computing the k-fold evaluation on a series of lambdas. 
Useful for estimating the appropriate regularization.
"""
from typing import Optional, List, Dict, Tuple, Iterable
from itertools import product, chain
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch

from .model import L0SparseCCA

def sparse_cca_kfold_cv(X1: np.ndarray, X2: np.ndarray, lambdaxs: List[float], lambdays: List[float], K: Optional[int] = 2, 
        priors: Optional[Tuple[np.ndarray, np.ndarray]] = None, sigma: Optional[float] = 1e-1,
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> Tuple[Tuple[float, float], Dict[Tuple[float,float],float], Dict[Tuple[float,float],float]]:
    """Evaluate multiple sparsity parameters to identify appropriate parameters.  

    Args:
        X1, X2: datasets for fitting sparse CCA models. Share first dimensions.
        lambdaxs, lambdays: sparsity parrameters applied to X1 and X2, respectively.
        K: number of folds to evaluate
        priors: priors over the computed canonical vectors. Estimated using CCA or PCA (see the fit function) 
        sigma: width of probability model used in reparametrization trick  
        dev: device on which the CCA model is trained
    
    Returns:
        Three results including the best regularization set and dictionaries encoding correleatins and l0 norms:
            A tuple showing the best tuple of regularization parameters
            A dictionary mapping tuples of regularization parameters to correlation 
            A dictionary mapping tuples of regularization parameters to l0 norm (loss function) 
    """
    ds = [X1.shape[1], X2.shape[1]]

    N = X1.shape[0]
    idxes = np.array_split(np.random.permutation(N), K)
    
    idx_groups = list()
    for Ki in range(K):
        train = sorted(chain(*[idxes[ii] for ii in range(K) if ii != Ki]))
        test = sorted(idxes[Ki])
        idx_groups.append((train, test))
    
    estimated_correlations = dict()
    l0_norms = dict()
    for lambdax, lambday in tqdm(product(lambdaxs, lambdays), total=len(lambdaxs) * len(lambdays)):
        valid_parameter_set = True

        _ls = [lambdax, lambday]
        _corrs, _norms = list(), list()
        for train, test in idx_groups:
            model = L0SparseCCA(C=1, lambdas=_ls, ds=ds, display=False, dev=dev)
            X1_train, X2_train = X1[train], X2[train]
            try:
                model.fit(X1_train,X2_train)
            except ValueError:
                print(f'The following regularization parameters result in invalid results: {lambdax} (input), {lambday} (embedding). Excluding.')
                valid_parameter_set = False
                break

            X1_test, X2_test = X1[test], X2[test]
            X1_tr, X2_tr = model.transform(X1_test, X2_test)
            _corr = np.sum(np.multiply(X1_tr, X2_tr)) / (np.linalg.norm(X1_tr) * np.linalg.norm(X2_tr))
            _corrs.append(_corr)
            
            a,b = model.canonical_weight_vectors
            l0 = np.count_nonzero(a) + np.count_nonzero(b)
            _norms.append(l0)
        
        if not valid_parameter_set:
            continue

        estimated_correlations[(lambdax, lambday)] = np.mean(_corrs)
        l0_norms[(lambdax, lambday)] = np.mean(_norms)
        
        print((lambdax, lambday), np.mean(_corrs), np.mean(_norms))

    best_reg = max(estimated_correlations.keys(), key=lambda t: estimated_correlations[t])
    return best_reg, estimated_correlations, l0_norms

def plot_kfold_regularization(correlations: Optional[dict] = None, norms: Optional[dict] = None, 
        figscale: int = 7, fpath: Optional[str] = None) -> plt.figure: 
    """A plot of the k-fold regularization paths.

    Args:
        correlations: a dictionary mapping pairs of regularization parameters to correlations
        norms: a dictionary mapping pairs of regularization parameters to l0 norms
        figscale: size of each subplot size
        fpath: optional filepath to save result to.

    Returns:
        A figure plotting the results of k-fold regularization
    """
    if correlations is None and norms is None:
        raise ValueError("No values passed for plotting")

    if correlations is not None and norms is not None:
        fig, axes = plt.subplots(1,2,figsize=(figscale*2,figscale))
        corr_ax = axes[0]
        norm_ax = axes[1]
    elif correlations is not None:
        fig, axes = plt.subplots(figsize=(figscale,figscale))
        corr_ax = axes
    else:
        fig, axes = plt.subplots(figsize=(figscale,figscale))
        norm_ax = axes
    
    if correlations is not None: _plot_grid_results(correlations, ax=corr_ax, corr_range = True)
    if norms is not None: _plot_grid_results(norms, ax=norm_ax, corr_range = False)

    if fpath:
        fig.savefig(f'{fpath}.svg', bbox_inches='tight', transparent=True, dpi=720)
        fig.savefig(f'{fpath}.png', bbox_inches='tight', transparent=True, dpi=720)
    return fig

def _plot_grid_results(args: dict, ax: plt.axis, corr_range: bool, corr_na: Optional[float] = 0, norm_na: Optional[int] = 0):
    """Modifies passed axis function with the associated grid plot.

    Args:
        args: dictionary of parameters mapped to a data function (correlation or norm)
        ax: subploto axis to plot values on
        corr_range: boolean defining if a correlation is passed; otherwise, assumes a norm is passed 
        corr_na: default value for correlations
        norm_na: default value for norms
    """
    xs = sorted(set(k[0] for k in args.keys()))
    ys = sorted(set(k[1] for k in args.keys()))
    
    Nx, Ny = len(xs), len(ys)
    
    xs_idxing = {xi:i for i, xi in enumerate(xs)}
    ys_idxing = {yi:(Ny - i - 1) for i, yi in enumerate(ys)}

    arr = np.empty((Nx, Ny))
    if corr_range:
        arr.fill(corr_na)
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bwr_gi', ['#1A99D5','#DCDCDC', '#DCDCDC', '#EE2B5E'])
    else:
        arr.fill(norm_na)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(args.values())) 
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('bwr_gi', ['#1A99D5', '#DCDCDC'])
   
    for (xi,yi),val in args.items():
        ri, ci = xs_idxing[xi], ys_idxing[yi]
        arr[ri,ci] = val
    
    formatter = (lambda s: f'{s:.3f}') if corr_range else (lambda s: str(int(s))) 
    im = ax.imshow(arr.T, cmap=cmap, norm=norm)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(("Correlation" if corr_range else "N"), rotation=-90, va="bottom")
    for (xi,yi),val in args.items():
        ri, ci = xs_idxing[xi], ys_idxing[yi]
        ax.text(ri,ci,formatter(val), ha='center', va='center', color='k') 

    ax.set_xticks(np.arange(Nx), xs, rotation=0)
    ax.set_yticks(np.arange(Ny), ys[::-1], rotation=0)
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(Nx)-.5, minor=True)
    ax.set_yticks(np.arange(Ny)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    ax.set_xlabel("Regularization (X)"); ax.set_ylabel("Regularization (Y)")

    if corr_range: ax.set_title("Correlation")
    else: ax.set_title("Non-zero (N)")
