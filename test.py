from transformers import pipeline
import re
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

import torch
from torch_geometric.data import Data
from torch.nn import Sequential as Seq, Linear, ReLU

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class DevGATConv(GATConv):
    def __init__(self, in_channels, out_channels):
        super(DevGATConv, self).__init__(in_channels, out_channels, aggr='add')  # "Add" aggregation (Step 5).


class DevGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DevGCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        print(deg)
        deg_inv_sqrt = deg.pow(-0.5)
        print(deg_inv_sqrt[row])
        print(deg_inv_sqrt[col])
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


if __name__ == '__main__':
    nodes = torch.tensor([[0.1, 0.3], [0.2, 0.2], [0.4, 0.1], [0.7, -0.1]])
    print(nodes)
    edges = torch.tensor([
        [0, 1], [1, 2], [0, 2], [1, 3], [2, 3]
    ]).permute(1, 0)

    undirected_edges = torch.tensor([
        [0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0], [1, 3], [3, 1], [3, 2], [2, 3]
    ]).permute(1, 0)
    conv = DevGCNConv(2, 2)

    feats = conv(nodes, edges)
    print(feats)
    feats2 = conv(nodes, undirected_edges)
    print(feats2)
