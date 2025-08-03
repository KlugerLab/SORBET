"""Class that fits a single component for sparse CCA analysis. Used for iteratively identifying CCA components in the model.py class.

For documentation, we consider CCA between two datasets, X and Y, of shared first dimension, (n x d1) and (n x d2), respectively.

Reimplements (and modifies for multiple components): 
    Lindenbaum, Salhov, Averbauch, Kluger. "L0-Sparse Canonical Correlation Analysis" ICLR (2023).
"""

from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from itertools import product, chain

import numpy as np
import torch
from torch.nn.parameter import Parameter

class SparseCCAComponentModel(torch.nn.Module):
    """PyTorch neural network for fitting single weight vectors 
    """
    def __init__(self, lambdas: Tuple[float, float], thetas: Tuple[np.ndarray, np.ndarray], 
            priors: Tuple[np.ndarray, np.ndarray], sigma: float, canonical_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
            device = None, dtype = None):
        """
        Args:
            lambdas: tuple of regularizaton parameters for the input and output values, respectively 
            thetas: means of each value used for the reparameterization trick
            priors: priors for these computed canonical vectors. 
            sigma: width of probability model used in reparametrization trick  
            canonical_vectors: previously computed canonical vectors. None if this is the first canonical vector.
                Necessary to compute orthogonalization penalty.
            device: device on which data is trained.
            dtype: type of passed data / parameters.
        """
        super().__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.lambdax, self.lambday = lambdas
        self.orth_lambda = max(max(lambdas), 1)

        self.mux = Parameter(torch.tensor(priors[0], **factory_kwargs), requires_grad=True) 
        self.muy = Parameter(torch.tensor(priors[1], **factory_kwargs), requires_grad=True)

        self.sigmax = torch.tensor(sigma * np.ones_like(priors[0]), requires_grad=False, **factory_kwargs)
        self.sigmay = torch.tensor(sigma * np.ones_like(priors[1]), requires_grad=False, **factory_kwargs)
        self.sigma = sigma
        
        self.thetax = Parameter(torch.tensor(thetas[0], **factory_kwargs), requires_grad=True)
        self.thetay = Parameter(torch.tensor(thetas[1], **factory_kwargs), requires_grad=True)
        
        self.orthogonalize = False
        if canonical_vectors:
            self.canonical_vectors_x = torch.tensor(canonical_vectors[0].T, requires_grad=False, **factory_kwargs)
            self.canonical_vectors_y = torch.tensor(canonical_vectors[1].T, requires_grad=False, **factory_kwargs)
            self.orthogonalize = True
      
        # Necessary to efficiently threshold; ugly:
        self.onesx = torch.tensor(np.ones_like(priors[0]), requires_grad=False, **factory_kwargs)
        self.zerosx = torch.tensor(np.zeros_like(priors[0]), requires_grad=False, **factory_kwargs)
        self.onesy = torch.tensor(np.ones_like(priors[1]), requires_grad=False, **factory_kwargs)
        self.zerosy = torch.tensor(np.zeros_like(priors[1]), requires_grad=False, **factory_kwargs)
    
    def sample_bernoulli(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples Bernoulli variables for reparamterization step 

        Returns:
            Thresholded variables for canonical vectors Bernoullis variables associated with X and Y, respectively. 
        """
        zx = torch.add(self.mux, torch.normal(self.zerosx, self.sigmax))
        zy = torch.add(self.muy, torch.normal(self.zerosy, self.sigmay))
        
        zx_thr = torch.max(self.zerosx, torch.min(zx, self.onesx))
        zy_thr = torch.max(self.zerosy, torch.min(zy, self.onesy))

        return zx_thr, zy_thr

    def forward(self, X: torch.tensor, Y: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes a single training step for estimating canonical vectors

        Args:
            X: an (n x d1) array of data 
            Y: an (n x d2) array of data 

        Returns:
            Three tensors encoding the correlation and (2) canonical vectors.  
        """
        zx, zy = self.sample_bernoulli()

        ai = torch.squeeze(torch.multiply(zx, self.thetax))
        Xi = torch.squeeze(torch.matmul(X, ai))
        Xi_norm = torch.linalg.vector_norm(Xi) 

        bi = torch.squeeze(torch.multiply(zy, self.thetay))
        Yi = torch.squeeze(torch.matmul(Y, bi))
        Yi_norm = torch.linalg.vector_norm(Yi) 

        corr = torch.divide(torch.sum(torch.multiply(Xi, Yi)), torch.multiply(Xi_norm, Yi_norm))

        return corr, ai, bi
    
    def get_loss_function(self, corr: torch.tensor, ai: torch.tensor, bi: torch.tensor) -> torch.Tensor:
        """Computes loss function for single vector model.

        Args:
            corr: current correlation estimate
            ai: length d1 canonical vecctors associated with the X dataset
            bi: length d2 canonical vecctors associated with the Y dataset

        Returns:
            The loss function evaluated at the current iteration.
        """
        # Sparsity Penalty: 
        sparse_pen = torch.multiply(torch.sum(0.5 - torch.multiply(torch.special.erf(-1 * torch.divide(self.mux, (np.sqrt(2) * self.sigma))), 0.5)), self.lambdax)
        sparse_pen += torch.multiply(torch.sum(0.5 - torch.multiply(torch.special.erf(-1 * torch.divide(self.muy, (np.sqrt(2) * self.sigma))), 0.5)), self.lambday) 

        # Orthogonal regularizer:
        if self.orthogonalize:
            orthogonal_pen = torch.multiply(torch.sum(torch.abs(torch.matmul(ai, self.canonical_vectors_x))), self.orth_lambda)
            orthogonal_pen += torch.multiply(torch.sum(torch.abs(torch.matmul(bi, self.canonical_vectors_y))), self.orth_lambda)
        else:
            orthogonal_pen = 0
        
        loss_fn = torch.multiply(-1, corr) + sparse_pen + orthogonal_pen
        return loss_fn

    def get_canonical_weight_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Getter function to extract the currently estimated canonical vectors. 

        Returns:
            Two array corresponding to the canonical vectors of the X and Y datasets. Sizes d1 and d2, respectively.
        """
        mux = self.mux.detach().cpu().numpy()
        muy = self.muy.detach().cpu().numpy()

        mux_thr = np.maximum(np.zeros_like(mux), np.minimum(np.ones_like(mux), mux))
        muy_thr = np.maximum(np.zeros_like(muy), np.minimum(np.ones_like(muy), muy))

        ai = np.multiply(mux_thr, self.thetax.detach().cpu().numpy())
        bi = np.multiply(muy_thr, self.thetay.detach().cpu().numpy())

        return ai, bi
    
    def get_model_correlation(self, X: torch.tensor, Y: torch.tensor) -> torch.Tensor:
        """Estimate correlation function under computed 

        Args:
            X: an (n x d1) array of data 
            Y: an (n x d2) array of data 

        Returns:
            Correlation value between dataset for the currently estimated canonical vectors.
        """
        corr, _, _ = self.forward(X, Y)
        return corr.item()
