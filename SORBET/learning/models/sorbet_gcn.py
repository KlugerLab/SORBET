"""Implementation of the GCN structure detailed in the SORBET paper. 

Note: Please cross reference base.py for full documentation. 
"""
from typing import List, Dict, Union, Optional
from enum import Enum
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Linear, Sigmoid, ReLU, ModuleList, Dropout, BatchNorm1d, BCEWithLogitsLoss, LayerNorm 
from torch_geometric.nn import GCNConv, MeanAggregation, pool, norm, DeepGCNLayer

from .model_specification import StructureSpecification 
from .base import BaseGraphModel 
from ._utils import parse_pooling_function 

# NOTE: Please see model spec for a description of how these arguments correspond to model inputs.
_VanillaSORBETSpecification = StructureSpecification(
        model_serializable_parameters = ["in_linear_channels", "conv_channels", "embedding_linear_channels", "out_linear_channels"],
        model_non_serializable_parameters = [("dropout", float, False), ("pooling_fn", "choice", False), ("block", "choice", False), 
            ("l1_penalty", float, True), ("topk_pooling", int, False)],
        model_input_specifier = "in_channel",
        random_seed_specifier = "random_seed"
        )

class GCNSorbetBase(BaseGraphModel):
    """GCN model with the following structure:
        - Linear Layers (`in_linear_channels`)
        - GCN Conv Layers (`conv_channels`)
        - Linear Layers (`embedding_linear_channels`)
        - Pooling Layer (default: global_mean)
        - Linear Layers (`out_linear_channels`)
    """
    
    def __init__(self, 
            in_channel: int,
            in_linear_channels: List[int],
            conv_channels: List[int],
            embedding_linear_channels: List[int],
            out_linear_channels: List[int],
            block: str = 'dense', dropout: float = 0.3, batch_norm_args: dict = {'eps': 1e-5, 'momentum': 0.1},
            pooling_fn: pool = "global_mean", topk_pooling: int = None, 
            loss_fn: torch.nn = BCEWithLogitsLoss, l1_penalty: Union[float, None] = None,
            random_seed: Optional[int] = None
            ):
        """
        Args:
            in_channel: size of input data (features)
            in_linear_channels: list of linear layer widths (`Input Layers`)
            conv_channels: list of conv layer widths (`Conv Layers`)
                block: block type, associated with the CGNConv layer in torch_geomtric.nn
            embedding_linear_channels: list of linear layer widths (`Embedding Layers`)
            out_linear_channels: list of linear layer widths (`Output Layers`)
            pooling_fn: the chosen pooling function. Applied between embedding_linear_channels and out_linear_channels.
                topk_pooling: boolean defining if the pooling function uses top-k pooling
            # Hyperparameters:
                dropout: dropout (applied to all layers)
                batch_norm_args: dictionary passed to batch normalization layer
            # Loss function choice: 
                loss_fn: a chosen loss function (def: torch.nn.BCEWithLogitsLoss)
                l1_penalty: define if an L1 penalty should be applied. Ignored if None or 0.
            random_seed: integer setting a random seed.
        """

        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.in_linear_layers = ModuleList()
        for i_channel, o_channel in zip([in_channel, *in_linear_channels], in_linear_channels):
            self.in_linear_layers.extend([
                Dropout(p = dropout), 
                Linear(i_channel, o_channel),
                LayerNorm(o_channel),
                ReLU()
                ])
        
        self.conv_layers = ModuleList()
        if block == 'dense':
            skip_size = 0
            for i_channel, o_channel, _ in zip([in_linear_channels[-1], *conv_channels], conv_channels, [0, *conv_channels]):
                conv_layer= GCNConv(i_channel + skip_size, o_channel)
                norm_layer = LayerNorm(o_channel)
                act_fn = ReLU()
                layer = DeepGCNLayer(conv_layer, norm_layer, act_fn, block, dropout)
                self.conv_layers.append(layer)
                skip_size += i_channel
            
            last_layer = skip_size + conv_channels[-1]
        elif block in {'res', 'res+', 'plain'}:
            for i_channel, o_channel, _ in zip([in_linear_channels[-1], *conv_channels], conv_channels, [0, *conv_channels]):
                conv_layer = GCNConv(i_channel, o_channel)
                norm_layer = LayerNorm(o_channel)
                act_fn = ReLU()
                layer = DeepGCNLayer(conv_layer, norm_layer, act_fn, block, dropout)
                self.conv_layers.append(layer)

            last_layer = conv_channels[-1]
        else:
            raise ValueError("Block description not handled")

        self.embed_linear_layers = ModuleList()
        for i_channel, o_channel in zip([last_layer, *embedding_linear_channels], embedding_linear_channels):
            lin_layer = Linear(i_channel, o_channel)
            self.embed_linear_layers.extend([
                Dropout(p = dropout),
                Linear(i_channel, o_channel),
                LayerNorm(o_channel),
                ReLU()
                ])
        
        if topk_pooling is not None and topk_pooling > 0:
            self.use_topk_pooling = True
            self.topk_pooler = pool.TopKPooling(in_channels=embedding_linear_channels[-1], ratio=topk_pooling)
        else:
            self.use_topk_pooling = False

        self.pooler = parse_pooling_function(pooling_fn) 
        
        self.out_linear_layers = ModuleList()
        last_layer = embedding_linear_channels[-1]
        for i_channel, o_channel in zip([last_layer, *out_linear_channels], [*out_linear_channels, 1]):
            lin_layer = Linear(i_channel, o_channel)
            self.out_linear_layers.append(lin_layer)
        
        self.output_nonlin = Sigmoid()
 
        # Initialize loss function
        self.loss_function = loss_fn()
        self.l1_penalty = l1_penalty if l1_penalty is not None else 0

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """See base.py for specification
        """
        x = self.get_cell_embedding(x, edge_index, batch)     
        x = self.get_subgraph_embedding(x, edge_index, batch)

        for ll in self.out_linear_layers:
            x = ll(x)
        
        return x
    
    def predict(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """See base.py for specification
        """
        x = self.forward(x, edge_index, batch)
        return self.output_nonlin(x)

    def get_loss_function(self, predictions: Tensor, labels: Tensor) -> Tensor:
        """See base.py for specification
        """
        loss = self.loss_function(predictions, labels)

        if self.l1_penalty > 0: 
            for p in self.parameters():
                loss += self.l1_penalty * torch.linalg.norm(torch.flatten(p), 1)
        
        return loss
    
    def get_cell_embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """See base.py for specification
        """
        for ll in self.in_linear_layers:
            x = ll(x)

        for conv in self.conv_layers:
            x = conv(x, edge_index = edge_index)

        for ll in self.embed_linear_layers:
            x = ll(x)
        
        return x
    
    def get_subgraph_embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """See base.py for specification
        """
        if self.use_topk_pooling:
            x, edge_index, _, batch, _, _ = self.topk_pooler(x, edge_index, batch=batch)
        return self.pooler(x, batch)

    @staticmethod
    def get_model_specification() -> StructureSpecification:
        """See base.py for specification
        """
        return _VanillaSORBETSpecification
