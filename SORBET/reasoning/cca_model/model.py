"""Wrapper method for estimating Sparse CCA embedding. 
Makes calls to _singlevector_cca.py for estimating each individual canonical vector.

For documentation, we consider CCA between two datasets, X and Y, of shared first dimension, (n x d1) and (n x d2), respectively.

Reimplements (and modifies for multiple components): 
    Lindenbaum, Salhov, Averbauch, Kluger. "L0-Sparse Canonical Correlation Analysis" ICLR (2023).
"""
import pickle
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from itertools import product, chain
from tqdm import tqdm as tqdm

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from ._data import CCAData
from ._singlevector_cca import SparseCCAComponentModel 

class L0SparseCCA:
    """Class for estimating the Sparse CCA method.
    """
    def __init__(self, C: Optional[int], lambdas: Tuple[float, float], ds: Tuple[int, int], 
            priors: Optional[Tuple[np.ndarray, np.ndarray]] = None, sigma: Optional[float] = 1e-1, r: Optional[float] = 0.25,
            optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam, display: Optional[bool] = True, correlation_threshold: Optional[int] = None, 
            learning_rate: Optional[float] = 1e-3, max_iters: Optional[int] = 250, tol: Optional[float] = 1e-10, batch_size: Optional[int] = 4096,
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ):
        """
        Args:
            C: number of canonical vectors to compute for each value
            lambdas: tuple of regularizaton parameters for the input and output values, respectively 
            ds: tuple of dimensions of the input data
            priors: priors over the computed canonical vectors. Estimated using CCA or PCA (see the fit function) 
            sigma: width of probability model used in reparametrization trick  
            r: sparsity fraction parameter applied to priors  
            optimizer: choice of optimizer used to train moodels
            display: display training progressbar 
            correlation_threshold: minimum threshold for correlation to reach
            learning_rate: step size for each training step
            max_iters: maximum number of training iterations
            tol: tolerance for training loss changes
            batch_size: batch size used in training model 
            dev: device on which the CCA model is trained
        """
        # Canonical Vectors Initialization:
        self.C = C
        self.canonical_weight_vectors = None
        
        # Model Prior Initialization:
        self.d1, self.d2 = ds
        self.priors_in = priors
        self.r = r * 100

        self.lambdas = lambdas
        self.sigma = sigma

        # Model Learning / Fitting Parameters:
        self.optimizer_class = optimizer
        self.LR = learning_rate
        self.maxiters = max_iters
        self.tol = tol
        self.batch_size = batch_size
        self.dev = dev
        self.display = display
        
        # Model Fitting Checks:
        self.losses = dict() 
        self.correlations = dict() 
        self.correlation_threshold = correlation_threshold if correlation_threshold else -1

    def _fit_canonical_vector(self, X: np.ndarray, Y: np.ndarray, thetas: Tuple[np.ndarray, np.ndarray], priors: Tuple[np.ndarray, np.ndarray]
            ) -> Tuple[SparseCCAComponentModel, List[List[float]], List[List[float]]]:
        """A function called by the fit function to iteratively fit canonical vectors.

        Args:
            X: an (n x d1) array of data 
            Y: an (n x d2) array of data 
            priors: Priors on the canonical vectors

        Returns: 
            Three lists corresponding to the training results including: 
                A single component model (see _singlevector_cca.py).
                A list of loss values. Length corresponds to number of epochs.
                A list of correlation values. Length corresponds to number of epochs.
        """
        model = SparseCCAComponentModel(self.lambdas, thetas, priors, self.sigma, canonical_vectors=self.canonical_weight_vectors, 
                device=self.dev, dtype=float) 
        model.to(self.dev)
        
        cca_data = CCAData(X.T,Y.T)
        data_loader = DataLoader(cca_data, batch_size=self.batch_size, shuffle=True)
     
        optimizer = self.optimizer_class(model.parameters(), lr=self.LR)
        
        Cindex = (self.canonical_weight_vectors[0].shape[0] if self.canonical_weight_vectors else 0) + 1

        losses, corrs = list(), list()
        last_loss = np.inf 
        for i in tqdm(range(self.maxiters), desc=f'Fitting C{Cindex}', disable=(not self.display)):
            _losses, _corrs = list(), list()
            
            for Xb, Yb in data_loader:
                Xb, Yb = Xb.to(self.dev), Yb.to(self.dev)
                
                corr, ai, bi = model(Xb, Yb)
                loss_fn = model.get_loss_function(corr, ai, bi)
                
                loss_fn.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                _losses.append(loss_fn.item())
                _corrs.append(corr.item())

            losses.append(_losses)
            corrs.append(_corrs)
            
            curr_loss = np.mean(losses[-1])
            if abs((last_loss -  curr_loss) / curr_loss) <= self.tol:
                break
            if np.isnan(curr_loss):
                break
            last_loss = curr_loss

        return model, losses, corrs

    def _update_canonical_vector(self, ai: np.ndarray, bi: np.ndarray, Ci: int):
        """
        A setter method for updating the saved canonical vectors

        Args:
            ai: a length d1 array representing the new canonical vectors associated to the X1 data
            bi: a length d2 array representing the new canonical vectors associated to the X2 data
            Ci: the index of the saved canonical vectors (zero-indexed) 
        """
        if self.canonical_weight_vectors is None:
            self.canonical_weight_vectors = (ai[np.newaxis,:], bi[np.newaxis,:])
        else:
            a, b = self.canonical_weight_vectors
            a, b = np.vstack([a, ai]), np.vstack([b, bi])

            assert a.shape[0] == (Ci + 1)
            assert b.shape[0] == (Ci + 1)
            
            self.canonical_weight_vectors = (a, b)
    
    def fit(self, X, Y):
        """Fits the model over the chosen number of components C (set at initialization)

        Args:
            X: an (d1 x n) dataset
            Y: an (d2 x n) dataset
        """
        try:
            standard_cca = CCA(n_components=self.C)
            standard_cca.fit(X,Y)
            standard_weights = [standard_cca.x_weights_, standard_cca.y_weights_]
        except ValueError:
            pca_in, pca_out = PCA(n_components=self.C), PCA(n_components=self.C)
            standard_weights = [pca_in.fit_transform(X.T), pca_out.fit_transform(Y.T)]
        
        if self.priors_in:
            priors = self.priors_in
        else:
            try:
                priors_cov = (X.T @ Y) / (X.shape[0] - 1)
                thr = np.percentile(priors_cov, self.r)
                priors_cov[priors_cov < thr] = 0

                U,_,Vh = np.linalg.svd(priors_cov)
                u,v = U[0], Vh.T[0]

                u[u < np.percentile(u, self.r)] = 0
                v[v < np.percentile(v, self.r)] = 0
                priors = u + 0.5, v + 0.5
            except ValueError:
                pca_in, pca_out = PCA(n_components=self.C), PCA(n_components=self.C)
                standard_weights = [pca_in.fit_transform(X)[:,0], pca_out.fit_transform(Y)[:,0]]
        
        
        for Ci in range(self.C):
            thetas = [standard_weights[0][:,Ci], standard_weights[1][:,Ci]]
            model, losses, corrs = self._fit_canonical_vector(X, Y, thetas, priors)
            
            last_corr = np.mean(corrs[-1])
            last_loss = np.mean(losses[-1])
            if np.isnan(last_loss) and self.C == 1:
                raise ValueError("Invalid hyperparameter resulting in null values")
            elif np.isnan(last_loss) or self.correlation_threshold > last_corr:
                break 
            
            # Update Model:
            ai, bi = model.get_canonical_weight_vectors()
            self._update_canonical_vector(ai, bi, Ci)
            
            self.losses[Ci] = losses
            self.correlations[Ci] = corrs

    def transform(self, X, Y) -> Tuple[np.ndarray, np.ndarray]:
        """A transform method similar to sklearn's transform methods.
        Transforms the inputs (X and Y) into the shared C-dimensional space.

        NOTE: Requires a previous call to a fit method to have a trained model.

        Args:
            X: an (d1 x n) dataset
            Y: an (d2 x n) dataset

        Returns:
            Two (C x n) matrices transformed into the shared, canonical space. Associated with X and Y, respectively.
        """
        if self.canonical_weight_vectors is None:
            raise ValueError("Attempting to transform inputs prior to fitting model. Default canonical weight vectors are not assumed.")
        
        a,b = self.canonical_weight_vectors
        Xt = X @ a.T
        Yt = Y @ b.T

        return Xt, Yt
       
    def fit_transform(self, X, Y) -> Tuple[np.ndarray, np.ndarray]:
        """A fit transform method similar to sklearn's fit_transform methods.

        Args:
            X: an (d1 x n) dataset
            Y: an (d2 x n) dataset

        Returns:
            Two (C x n) matrices transformed into the shared, canonical space. Associated with X and Y, respectively.
        """
        self.fit(X,Y)
        return self.transform(X,Y)
    
    def plot_fit(self, figsize: Optional[Tuple[int,int]] = (10,5), fpath: Optional[str] = None, axes: Optional[plt.axis] = None) -> Optional[plt.figure]:
        """Plots the fit of the CCA model. 
        Requires two accessible axes for the loss function and correlation function.

        Args:
            figsize: An optional argument defining the size of a figure. Ignored if axes is passed.
            fpath: An optional argument defining a location to save figures to. 
            axes: List of two axes to plot the loss and correlation functions, respectively.
        
        Returns:
            Returns the created figure if axes is not passed. Otherwise, no returned value.
        """
        if axes is None:
            fig, axes = plt.subplots(1,2,figsize=figsize)
            return_fig = True
        else:
            return_fig = False

        emax = None
        # Plot Loss Function:
        for ci, li in sorted(self.losses.items(), key=lambda t: t[0]):
            _ls = np.array(li)
            _ls_means, _ls_stds = np.mean(_ls, axis=1), np.std(_ls, axis=1)
            xs = np.arange(1, _ls.shape[0]+1)

            axes[0].errorbar(xs, _ls_means, yerr=_ls_stds, label=str(ci))
        else: emax = _ls.shape[0]

        # Plot Correlation:
        for ci, li in sorted(self.correlations.items(), key=lambda t: t[0]):
            _cs = np.array(li)
            _cs_means, _cs_stds = np.mean(_cs, axis=1), np.std(_cs, axis=1)
            xs = np.arange(1, _ls.shape[0]+1)

            axes[1].errorbar(xs, _cs_means, yerr=_cs_stds, label=str(ci))
        
        epochs_space = np.arange(0, emax+5, 25) 

        axes[0].legend(loc='upper right', title="Component Index")
        axes[0].set_xticks(epochs_space)
        axes[0].grid(True)
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Function")

        axes[1].legend(loc='lower right', title="Component Index")
        axes[1].set_ylim(-0.025, 1.025)
        axes[1].set_yticks(np.linspace(0.0,1,11))
        axes[1].set_xticks(epochs_space)
        axes[1].grid(True)
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Correlation")
        axes[1].set_title("Correlation Function")

        if fpath is not None:
            fig.savefig(fpath, bbox_inches='tight', transparent=True, dpi=720)
        
        if return_fig:
            return fig

def save_cca_model(model: L0SparseCCA, fpath: str):
    """Saves a fit CCA model. Reload using load_cca_model.

    Args:
        model: Sparse CCA model to save.
        fpath: filepath where CCA model is to be saved
    """
    init_dict = {
                'C': model.C,
                'lambdas': model.lambdas, 
                'ds': [model.d1, model.d2] 
            }
    
    state_dict = model.__dict__
    
    with open(fpath, 'wb+') as ofile:
        pickle.dump([init_dict, state_dict], ofile)

def load_cca_model(fpath: str) -> L0SparseCCA:
    """Returns a previously fit CCA model. Save using save_cca_model.

    Args:
        fpath: file where CCA model was previously saved.

    Returns:
        Sparse CCA model at saved filepath. 
    """
    with open(fpath, 'rb') as ifile:
        init_args, state_dict = pickle.load(ifile)

    new_model = L0SparseCCA(**init_args)
    new_model.__dict__ = state_dict

    return new_model
