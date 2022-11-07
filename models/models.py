import numpy as np
import time
import math
import sys
from icecream import ic
import scipy.sparse
from tqdm import tqdm
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax, remove_self_loops, add_self_loops, \
    degree, remove_isolated_nodes, contains_isolated_nodes, dense_to_sparse
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Sequential as Seq, Linear, ReLU
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add, scatter, scatter_max
from torch_sparse import masked_select_nnz, SparseTensor, matmul
from torch_geometric.typing import (OptPairTensor, PairTensor, Adj, Size, NoneType, OptTensor)
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, AGNNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.init as init
from torch_geometric.nn import inits
from torch.nn.modules.module import Module

from utils.data_transform import dense_to_sparse_coo_tensor, edge_index_to_adj_mx, \
    row_normalize, sparse_mx_to_torch_sparse_tensor, edge_index_to_torch_coo_tensor

class SNGNN_Plus_Plus(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layers, top_k=2, thr=0.0,
                 init_beta=0.5, is_remove_self_loops=1, droput_rate=0.5, bn=False):
        super().__init__()
        self.top_k = top_k
        self.thr = thr  # threshold
        self.bn = bn
        self.init_beta = init_beta
        self.num_nodes = num_nodes
        if is_remove_self_loops == 1:
            self.is_remove_self_loops = True
        else:
            self.is_remove_self_loops = False
        self.lins = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(SNConv_plus_plus(in_channels, out_channels, self.num_nodes, self.top_k, self.thr,
                                           self.init_beta, self.is_remove_self_loops, self.bn))
        else:
            self.lins.append(SNConv_plus_plus(in_channels, hidden_channels, self.num_nodes, self.top_k, self.thr,
                                           self.init_beta, self.is_remove_self_loops, self.bn))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(SNConv_plus_plus(hidden_channels, hidden_channels, self.num_nodes, self.top_k, self.thr,
                                               self.init_beta, self.is_remove_self_loops, self.bn))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(SNConv_plus_plus(hidden_channels, out_channels,  self.num_nodes, self.top_k, self.thr,
                                           self.init_beta, self.is_remove_self_loops, self.bn))
        self.dropout = torch.nn.Dropout(p=droput_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x, edge_index)
            x = F.relu(x, inplace=True)
            if self.bn:
                x = self.bns[i](x)
            x = self.dropout(x)
        x = self.lins[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class SNConv_plus_plus(MessagePassing):
    def __init__(self, in_channels, out_channels, num_nodes, top_k=2, thr=0.0, init_beta=0.5,
                 is_remove_self_loops=True, bias: bool = False, aggr='mean'):
        super().__init__(aggr=aggr)
        self.top_k = top_k
        self.thr = thr  # threshold
        self.w = torch.nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes
        self.is_remove_self_loops = is_remove_self_loops
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.beta = Parameter(torch.Tensor(1))
        # self.beta = 0.0
        # self.beta = Parameter(torch.Tensor(out_channels))
        self.init_beta = init_beta
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self.w.reset_parameters()
        # self.beta.data.fill_(0.5)
        self.beta.data.fill_(self.init_beta)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print(contains_isolated_nodes(edge_index))
        if self.is_remove_self_loops:
            edge_index, _ = remove_self_loops(edge_index, None)
        x = self.lin(x)
        norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min()  # for sampling
            adj = SparseTensor(row=row, col=col, sparse_sizes=(
                self.num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            adj = edge_index.to_torch_sparse_coo_tensor()
        out_0 = self.w(adj)
        # edge_index = edge_index_sim
        out_1 = self.propagate(edge_index, x=x, norm=norm)
        # print(self.beta)
        out = self.beta*out_0 + (1-self.beta)*out_1
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_i, x_j, norm_i, norm_j, x, edge_index, index):
        norm = (norm_i * norm_j).sum(dim=-1)  # [E, ]
        tmp_weight = norm.clone()
        # dtype = norm.dtype, device = norm.device
        weight = torch.zeros(norm.shape, dtype=norm.dtype, device=norm.device)
        max_indexes = []
        for i in range(self.top_k):
            # index: [E, 1] tmp_weight->norm:
            max_weight, max_index = scatter_max(tmp_weight, index, dim=0)
            # # not exist node in max_index possess the val of index.shape,
            # because of isolated node or undirected edge.
            new_max_weight = torch.where(max_index == tmp_weight.shape[0],
                                         torch.full_like(max_weight, -2), max_weight)
            new_max_index = max_index[torch.where(new_max_weight >= self.thr)[0]]
            tmp_weight = tmp_weight.scatter(-1, new_max_index, -1.1)
            max_indexes.append(new_max_index)
        for i in range(self.top_k):
            weight = weight.scatter(-1, max_indexes[i], norm[max_indexes[i]])
        tmp = weight.view(-1, 1) * x_j
        return tmp


class SNGNN_Plus(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, num_layers, top_k=2, thr=0.0,
                 is_remove_self_loops=1, droput_rate=0.5, bn=False):
        super().__init__()
        self.top_k = top_k
        self.thr = thr  # threshold
        self.bn = bn
        self.num_nodes = num_nodes
        if is_remove_self_loops == 1:
            self.is_remove_self_loops = True
        else:
            self.is_remove_self_loops = False
        self.lins = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(SNConv_plus(in_channels, out_channels, self.num_nodes, self.top_k, self.thr,
                                           self.is_remove_self_loops, self.bn))
        else:
            self.lins.append(SNConv_plus(in_channels, hidden_channels, self.num_nodes, self.top_k, self.thr,
                                           self.is_remove_self_loops, self.bn))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(SNConv_plus(hidden_channels, hidden_channels, self.num_nodes, self.top_k, self.thr,
                                               self.is_remove_self_loops, self.bn))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(SNConv_plus(hidden_channels, out_channels,  self.num_nodes, self.top_k, self.thr,
                                           self.is_remove_self_loops, self.bn))
        self.dropout = torch.nn.Dropout(p=droput_rate)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x, edge_index)
            x = F.relu(x, inplace=True)
            if self.bn:
                x = self.bns[i](x)
            x = self.dropout(x)
        x = self.lins[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class SNConv_plus(MessagePassing):
    def __init__(self, in_channels, out_channels, num_nodes, top_k=2, thr=0.0,
                 is_remove_self_loops=True, bias: bool = False, aggr='mean'):
        super().__init__(aggr=aggr)
        self.top_k = top_k
        self.thr = thr  # threshold
        self.num_nodes = num_nodes
        self.is_remove_self_loops = is_remove_self_loops
        self.lin = torch.nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.is_remove_self_loops:
            edge_index, _ = remove_self_loops(edge_index, None)
        x = self.lin(x)
        norm = F.normalize(x, p=2., dim=-1)
        out = self.propagate(edge_index, x=x, norm=norm)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_i, x_j, norm_i, norm_j, x, edge_index, index):
        norm = (norm_i * norm_j).sum(dim=-1)  # [E, ]
        tmp_weight = norm.clone()
        # dtype = norm.dtype, device = norm.device
        weight = torch.zeros(norm.shape, dtype=norm.dtype, device=norm.device)
        max_indexes = []
        for i in range(self.top_k):
            # index: [E, 1] tmp_weight->norm:
            max_weight, max_index = scatter_max(tmp_weight, index, dim=0)
            # # not exist node in max_index possess the val of index.shape,
            # because of isolated node or undirected edge.
            new_max_weight = torch.where(max_index == tmp_weight.shape[0],
                                         torch.full_like(max_weight, -2), max_weight)
            new_max_index = max_index[torch.where(new_max_weight >= self.thr)[0]]
            tmp_weight = tmp_weight.scatter(-1, new_max_index, -1.1)
            max_indexes.append(new_max_index)
        for i in range(self.top_k):
            weight = weight.scatter(-1, max_indexes[i], norm[max_indexes[i]])
        tmp = weight.view(-1, 1) * x_j
        return tmp

class SNGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, bn=False):
        super().__init__()
        self.bn = bn
        self.lins = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(SNConv(in_channels, out_channels))
        else:
            self.lins.append(SNConv(in_channels, hidden_channels))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(SNConv(hidden_channels, hidden_channels))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(SNConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x, edge_index)
            x = F.relu(x, inplace=True)
            if self.bn:
                x = self.bns[i](x)
            x = self.dropout(x)
        x = self.lins[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class SNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean', bias: bool = True,):
        super().__init__(aggr=aggr)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        norm = F.normalize(x, p=2., dim=-1)
        out = self.propagate(edge_index, x=x, norm=norm)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_i, x_j, norm_i, norm_j, x, edge_index):
        norm = (norm_i * norm_j).sum(dim=-1)
        tmp = norm.view(-1, 1) * x_j
        return tmp

class AGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, bn=False):
        super().__init__()
        self.bn = bn
        self.lins = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(AGNNConv(in_channels, out_channels))
        else:
            self.lins.append(AGNNConv(in_channels, hidden_channels))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(AGNNConv(hidden_channels, hidden_channels))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(AGNNConv(hidden_channels, out_channels))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x, edge_index)
            x = F.relu(x, inplace=True)
            if self.bn:
                x = self.bns[i](x)
            x = self.dropout(x)
        x = self.lins[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class AGNNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, aggr='add', add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.add_self_loops = add_self_loops
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))
        x = self.lin(x)
        x_norm = F.normalize(x, p=2., dim=-1)

        # propagate_type: (x: Tensor, x_norm: Tensor)
        return self.propagate(edge_index, x=x, x_norm=x_norm, size=None)

    def message(self, x_j, x_norm_i, x_norm_j,
                index, ptr: OptTensor,
                size_i: Optional[int]):
        alpha = (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha.view(-1, 1)

class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, data):
        # N = extra_data['num_nodes']
        # edge_index = data['edge_index']
        edge_index = data.edge_index
        # edge_index = extra_data['adj']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min()  # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(
                self.num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        # return logits
        return F.log_softmax(logits, dim=1)


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
    # def forward(self, data):
        if not input_tensor:
            x = data.x
        else:
            x = data
        # x = data.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # edge_index = data.graph['edge_index']
        # x = data.graph['node_feat']
        x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ lower memory version (if out_channels < in_channels)
        takes weight multiplication first, then propagate
        takes hops power of the normalized adjacency
        """
        super(SGCMem, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # edge_index = data.graph['edge_index']
        # x = data.graph['node_feat']
        x = self.lin(x)
        ######
        n = data.graph['num_nodes']
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for _ in range(self.hops):
            x = matmul(adj_t, x)

        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        # return x
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu
        self.sampling = sampling
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, adjs=None, x_batch=None):
        if not self.sampling:
            x, edge_index = data.x, data.edge_index
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)



class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """

    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin  # handle multiple binary tasks

    def forward(self, data, train_idx):
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(
                row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight = None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if data.label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(
                data.label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2*self.out_channels)).to(adj_t.device())
            for task in range(data.label.shape[1]):
                y[train_idx, 2*task:2*task +
                    2] = F.one_hot(data.label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = data.label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1-self.alpha)*y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(data.label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        return result


class MixHopLayer(nn.Module):
    """ Our MixHop layer """

    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x)]
        for j in range(1, self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        # x = data.graph['node_feat']
        # n = data.graph['num_nodes']
        # edge_index = data.graph['edge_index']
        x, n, edge_index = data.x, data.num_nodes, data.edge_index
        # print(type(edge_index))
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        # return x
        return F.log_softmax(x, dim=1)

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        cached = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(
            jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(
                hidden_channels * num_layers, out_channels)
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        # x = data.graph['node_feat']
        x, edge_index = data.x, data.edge_index
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            # x = conv(x, data.graph['edge_index'])
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, data.graph['edge_index'])
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        # return x
        return F.log_softmax(x, dim=1)


class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu  # note: uses elu

        self.jump = JumpingKnowledge(
            jk_type, channels=hidden_channels*heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(
                hidden_channels*heads*num_layers, out_channels)
        else:  # max or lstm
            self.final_project = nn.Linear(hidden_channels*heads, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        # x = data.graph['node_feat']
        xs = []
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        # return x
        return F.log_softmax(x, dim=1)


class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                                 hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(
                    hidden_channels*2*len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        # x = data.graph['node_feat']
        # n = data.graph['num_nodes']
        x, n = data.x, data.num_nodes
        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        # return x
        return F.log_softmax(x, dim=1)


class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, out_channels,
                       num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        # edge_index = data.graph['edge_index']
        x, edge_index = data.x, data.edge_index
        x = self.mlp(x, input_tensor=True)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            # return x
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            # return x
            return F.log_softmax(x, dim=1)

class LINK_Concat(nn.Module):
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):
        super(LINK_Concat, self).__init__()
        self.mlp = MLP(in_channels + num_nodes, hidden_channels,
                       out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.cache = cache
        self.x = None

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, data):
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
            # N = data.graph['num_nodes']
            # feat_dim = data.graph['node_feat']
            # row, col = data.graph['edge_index']
            N, feat_dim, row, col = data.num_nodes, data.x, data.edge_index[0], data.edge_index[1]
            col = col + self.in_channels
            feat_nz = feat_dim.nonzero(as_tuple=True)
            feat_row, feat_col = feat_nz
            full_row = torch.cat((feat_row, row))
            full_col = torch.cat((feat_col, col))
            value = feat_dim[feat_nz]
            full_value = torch.cat((value,
                                    torch.ones(row.shape[0], device=value.device)))
            x = SparseTensor(row=full_row, col=full_col,
                             sparse_sizes=(N, N+self.in_channels)
                             ).to_torch_sparse_coo_tensor()
            if self.cache:
                self.x = x
        else:
            x = self.x
        logits = self.mlp(x, input_tensor=True)
        # logits = self.mlp(x, input_tensor=True)
        # return logits
        return F.log_softmax(logits, dim=1)


class LINKX(nn.Module):
    """ our LINKX method with skip connections
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False,
                 inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()
        self.mlpA = MLP(num_nodes, hidden_channels,
                        hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels,
                        hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels,
                             out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, data):
        # m = data.graph['num_nodes']
        # feat_dim = data.graph['node_feat']
        # row, col = data.graph['edge_index']
        m, feat_dim, row, col = data.num_nodes, data.x, data.edge_index[0], data.edge_index[1]
        row = row-row.min()
        A = SparseTensor(row=row, col=col,
                         sparse_sizes=(m, self.num_nodes)
                         ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(feat_dim, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)
        # return x
        return F.log_softmax(x, dim=1)


class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1, Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, out_channels,
                       num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        # edge_index = data.graph['edge_index']
        x, edge_index = data.x, data.edge_index
        x = self.mlp(x, input_tensor=True)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            # return x
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            # return x
            return F.log_softmax(x, dim=1)


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        # x = data.graph['node_feat']
        # n = data.graph['num_nodes']
        # edge_index = data.graph['edge_index']
        x, n , edge_index = data.x, data.num_nodes, data.edge_index
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(
                row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        # return x
        return F.log_softmax(x, dim=1)


# extra_info
class MLPNORM(nn.Module):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta,
                 norm_func_id, norm_layers, orders, orders_func_id, device):
        super(MLPNORM, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nnodes, nhid)
        # self.bn1 = nn.BatchNorm1d(nhid)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.norm_layers = norm_layers
        self.orders = orders
        self.device = device
        self.class_eye = torch.eye(self.nclass).to(device)
        self.orders_weight = Parameter(
            (torch.ones(orders, 1) / orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(nclass, orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders).to(device), requires_grad=True
        )
        self.diag_weight = Parameter(
            (torch.ones(nclass, 1) / nclass).to(device), requires_grad=True
        )
        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.nclass, 1) / self.nclass).to(self.device), requires_grad=True
        )

    # def forward(self, x, adj):
    def forward(self, data, extra_info):
        x, adj = data.x, extra_info['adj_dense']
        # print(x.shape)
        # xd = F.dropout(x, self.dropout, training=self.training)
        # adjd = F.dropout(adj, self.dropout, training=self.training)
        xX = self.fc1(x)
        # x = self.bn1(x)
        xA = self.fc4(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        # print(x.shape)
        x = torch.squeeze(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        # x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        # print(x.shape)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        # return x
        return F.log_softmax(x, dim=1)

    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1.0 - self.gamma
        coe2 = 1.0 / coe1
        # print(x.shape)
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        res = torch.squeeze(res)
        return res

    def norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders


class GGCNlayer_SP(nn.Module):
    def __init__(self, in_features, out_features, device, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer_SP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        self.deg_intercept_init = deg_intercept_init
        self.scale_init = scale_init
        self.use_decay = use_decay
        self.device = device

        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5, 0.0]))
            else:
                self.deg_coeff = nn.Parameter(
                    torch.tensor([deg_intercept_init, 0.0]))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            self.adj_remove_diag = None
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(scale_init*torch.ones([1]))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)

    def reset_parameters(self):
        self.fcn.reset_parameters()

        if self.use_degree:
            if self.use_decay:
                self.deg_coeff = nn.Parameter(
                    torch.tensor([0.5, 0.0]).to(self.device))
            else:
                self.deg_coeff = nn.Parameter(
                    torch.tensor([self.deg_intercept_init, 0.0]).to(self.device))
        if self.use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]).to(self.device))
            if self.use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]).to(self.device))
            else:
                self.scale = nn.Parameter(
                    self.scale_init*torch.ones([1]).to(self.device))

    def precompute_adj_wo_diag(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        adj_wo_diag_ind = (adj_i[0, :] != adj_i[1, :])
        self.adj_remove_diag = torch.sparse.FloatTensor(
            adj_i[:, adj_wo_diag_ind], adj_v[adj_wo_diag_ind], adj.size())

    def non_linear_degree(self, a, b, s):
        i = s._indices()
        v = s._values()
        return torch.sparse.FloatTensor(i, self.sftpls(a*v+b), s.size())

    def get_sparse_att(self, adj, Wh):
        i = adj._indices()
        Wh_1 = Wh[i[0, :], :]
        Wh_2 = Wh[i[1, :], :]
        sim_vec = F.cosine_similarity(Wh_1, Wh_2)
        sim_vec_pos = F.relu(sim_vec)
        sim_vec_neg = -F.relu(-sim_vec)
        return torch.sparse.FloatTensor(i, sim_vec_pos, adj.size()), torch.sparse.FloatTensor(i, sim_vec_neg, adj.size())

    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.non_linear_degree(
                self.deg_coeff[0], self.deg_coeff[1], degree_precompute)

        Wh = self.fcn(h)
        if self.use_sign:
            if self.adj_remove_diag is None:
                self.precompute_adj_wo_diag(adj)
        if self.use_sign:
            e_pos, e_neg = self.get_sparse_att(adj, Wh)
            if self.use_degree:
                attention_pos = self.adj_remove_diag*sc*e_pos
                attention_neg = self.adj_remove_diag*sc*e_neg
            else:
                attention_pos = self.adj_remove_diag*e_pos
                attention_neg = self.adj_remove_diag*e_neg

            prop_pos = torch.sparse.mm(attention_pos, Wh)
            prop_neg = torch.sparse.mm(attention_neg, Wh)

            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.sparse.mm(adj*sc, Wh)
            else:
                prop = torch.sparse.mm(adj, Wh)

            result = prop
        return result


class GGCNlayer(nn.Module):
    def __init__(self, in_features, out_features, device, use_degree=True, use_sign=True, use_decay=True, scale_init=0.5, deg_intercept_init=0.5):
        super(GGCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fcn = nn.Linear(in_features, out_features)
        self.use_degree = use_degree
        self.use_sign = use_sign
        self.deg_intercept_init = deg_intercept_init
        self.scale_init = scale_init
        self.use_decay = use_decay
        self.device = device

        if use_degree:
            if use_decay:
                self.deg_coeff = nn.Parameter(
                    torch.tensor([0.5, 0.0]).to(self.device))
            else:
                self.deg_coeff = nn.Parameter(
                    torch.tensor([deg_intercept_init, 0.0]).to(self.device))
        if use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]).to(self.device))
            if use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]).to(self.device))
            else:
                self.scale = nn.Parameter(
                    scale_init*torch.ones([1]).to(self.device))
        self.sftmax = nn.Softmax(dim=-1)
        self.sftpls = nn.Softplus(beta=1)

    def reset_parameters(self):
        self.fcn.reset_parameters()

        if self.use_degree:
            if self.use_decay:
                self.deg_coeff = nn.Parameter(torch.tensor([0.5, 0.0]))
            else:
                self.deg_coeff = nn.Parameter(
                    torch.tensor([self.deg_intercept_init, 0.0]))
        if self.use_sign:
            self.coeff = nn.Parameter(0*torch.ones([3]))
            if self.use_decay:
                self.scale = nn.Parameter(2*torch.ones([1]))
            else:
                self.scale = nn.Parameter(self.scale_init*torch.ones([1]))

    def forward(self, h, adj, degree_precompute):
        if self.use_degree:
            sc = self.deg_coeff[0]*degree_precompute+self.deg_coeff[1]
            sc = self.sftpls(sc)

        Wh = self.fcn(h)
        if self.use_sign:
            prod = torch.matmul(Wh, torch.transpose(Wh, 0, 1))
            sq = torch.unsqueeze(torch.diag(prod), 1)
            scaling = torch.matmul(sq, torch.transpose(sq, 0, 1))
            e = prod/torch.max(torch.sqrt(scaling), 1e-9 *
                               torch.ones_like(scaling))
            e = e-torch.diag(torch.diag(e))
            if self.use_degree:
                attention = e*adj*sc
            else:
                attention = e*adj

            attention_pos = F.relu(attention)
            attention_neg = -F.relu(-attention)
            prop_pos = torch.matmul(attention_pos, Wh)
            prop_neg = torch.matmul(attention_neg, Wh)

            coeff = self.sftmax(self.coeff)
            scale = self.sftpls(self.scale)
            result = scale*(coeff[0]*prop_pos+coeff[1]*prop_neg+coeff[2]*Wh)

        else:
            if self.use_degree:
                prop = torch.matmul(adj*sc, Wh)
            else:
                prop = torch.matmul(adj, Wh)

            result = prop

        return result

# extra_info
class GGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, decay_rate, exponent, device, use_degree=True, use_sign=True, use_decay=True, use_sparse=False, scale_init=0.5, deg_intercept_init=0.5, use_bn=False, use_ln=False):
        super(GGCN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if use_sparse:
            model_sel = GGCNlayer_SP
        else:
            model_sel = GGCNlayer
        self.convs.append(model_sel(nfeat, nhidden, device, use_degree,
                          use_sign, use_decay, scale_init, deg_intercept_init))
        for _ in range(nlayers-2):
            self.convs.append(model_sel(nhidden, nhidden, device, use_degree,
                              use_sign, use_decay, scale_init, deg_intercept_init))
        self.convs.append(model_sel(nhidden, nclass, device, use_degree,
                          use_sign, use_decay, scale_init, deg_intercept_init))
        self.fcn = nn.Linear(nfeat, nhidden)
        self.act_fn = F.elu
        self.dropout = dropout
        self.use_decay = use_decay
        if self.use_decay:
            self.decay = decay_rate
            self.exponent = exponent
        self.degree_precompute = None
        self.use_degree = use_degree
        self.use_sparse = use_sparse
        self.use_norm = use_bn or use_ln
        if self.use_norm:
            self.norms = nn.ModuleList()
        if use_bn:
            for _ in range(nlayers-1):
                self.norms.append(nn.BatchNorm1d(nhidden))
        if use_ln:
            for _ in range(nlayers-1):
                self.norms.append(nn.LayerNorm(nhidden))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fcn.reset_parameters()

        if self.use_norm:
            for norm in self.norms:
                norm.reset_parameters()

    def precompute_degree_d(self, adj):
        diag_adj = torch.diag(adj)
        diag_adj = torch.unsqueeze(diag_adj, dim=1)
        self.degree_precompute = diag_adj / \
            torch.max(adj, 1e-9*torch.ones_like(adj))-1

    def precompute_degree_s(self, adj):
        adj_i = adj._indices()
        adj_v = adj._values()
        # print('adj_i', adj_i.shape)
        # print(adj_i)
        # print('adj_v', adj_v.shape)
        # print(adj_v)
        adj_diag_ind = (adj_i[0, :] == adj_i[1, :])
        adj_diag = adj_v[adj_diag_ind]
        # print(adj_diag)
        # print(adj_diag[0])
        v_new = torch.zeros_like(adj_v)
        for i in tqdm(range(adj_i.shape[1])):
            # print('adj_i[0,', i, ']', adj_i[0, i])
            v_new[i] = adj_diag[adj_i[0, i]]/adj_v[i]-1
        self.degree_precompute = torch.sparse.FloatTensor(
            adj_i, v_new, adj.size())

    # def forward(self, x, adj):
    def forward(self, data, extra_info):
        x, adj = data.x, extra_info['adj_coo_tensor']
        # if self.use_degree:
        #     if self.degree_precompute is None:
        #         if self.use_sparse:
        #             self.precompute_degree_s(adj)
        #         else:
        #             self.precompute_degree_d(adj)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_previous = self.fcn(x)
        layer_previous = self.act_fn(layer_previous)
        layer_inner = self.convs[0](x, adj, self.degree_precompute)

        for i, con in enumerate(self.convs[1:]):
            if self.use_norm:
                layer_inner = self.norms[i](layer_inner)
            layer_inner = self.act_fn(layer_inner)
            layer_inner = F.dropout(
                layer_inner, self.dropout, training=self.training)
            if i == 0:
                layer_previous = layer_inner + layer_previous
            else:
                if self.use_decay:
                    coeff = math.log(self.decay/(i+2)**self.exponent+1)
                else:
                    coeff = 1
                layer_previous = coeff*layer_inner + layer_previous
            layer_inner = con(layer_previous, adj, self.degree_precompute)
        # return layer_inner
        return F.log_softmax(layer_inner, dim=1)


class GraphConvolution2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, model_type, output_layer=0, variant=False):
        super(GraphConvolution2, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type, self.variant = in_features, out_features, output_layer, model_type, variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features).cuda()), Parameter(
                torch.FloatTensor(in_features, out_features).cuda()), Parameter(torch.FloatTensor(in_features, out_features).cuda())
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1).cuda(
            )), Parameter(torch.FloatTensor(out_features, 1).cuda()), Parameter(torch.FloatTensor(out_features, 1).cuda())
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(1, 1).cuda(
            )), Parameter(torch.FloatTensor(1, 1).cuda()), Parameter(torch.FloatTensor(1, 1).cuda())

            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1)), Parameter(
                torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1))
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(
                1, 1)), Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1))

            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))

        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm(
            (output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)], 1)), self.att_vec)/T, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high):
        output = 0
        if self.model_type == 'mlp':
            output_mlp = (torch.mm(input, self.weight_mlp))
            return output_mlp
        elif self.model_type == 'sgc' or self.model_type == 'gcn':
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == 'acmgcn' or self.model_type == 'acmsnowball':
            if self.variant:
                output_low = (torch.spmm(adj_low, F.relu(
                    torch.mm(input, self.weight_low))))
                output_high = (torch.spmm(adj_high, F.relu(
                    torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))
            else:
                output_low = F.relu(torch.spmm(
                    adj_low, (torch.mm(input, self.weight_low))))
                output_high = F.relu(torch.spmm(
                    adj_high, (torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
        elif self.model_type == 'acmsgc':
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            # torch.mm(input, self.weight_high) - torch.spmm(self.A_EXP,  torch.mm(input, self.weight_high))
            output_high = torch.spmm(
                adj_high,  torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)

            # self.attention(F.relu(output_low), F.relu(output_high), F.relu(output_mlp))
            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# extra_info
class ACMGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, model_type, nlayers=1, variant=False):
        super(ACMGCN, self).__init__()
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.nlayers, = model_type, nlayers
        if self.model_type == 'mlp':
            self.gcns.append(GraphConvolution2(
                nfeat, nhid, model_type=model_type))
            self.gcns.append(GraphConvolution2(
                nhid, nclass, model_type=model_type, output_layer=1))
        elif self.model_type == 'gcn' or self.model_type == 'acmgcn':
            self.gcns.append(GraphConvolution2(
                nfeat, nhid,  model_type=model_type, variant=variant))
            self.gcns.append(GraphConvolution2(
                nhid, nclass,  model_type=model_type, output_layer=1, variant=variant))
        elif self.model_type == 'sgc' or self.model_type == 'acmsgc':
            self.gcns.append(GraphConvolution2(
                nfeat, nclass, model_type=model_type))
        elif self.model_type == 'acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolution2(
                    k * nhid + nfeat, nhid, model_type=model_type, variant=variant))
            self.gcns.append(GraphConvolution2(
                nlayers * nhid + nfeat, nclass, model_type=model_type, variant=variant))
        self.dropout = dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    # def forward(self, x, adj_low, adj_high):
    def forward(self, data, extra_info):
        x = data.x
        adj_low, adj_high = extra_info['adj_low'], extra_info['adj_high']
        if self.model_type == 'acmgcn' or self.model_type == 'acmsgc' or self.model_type == 'acmsnowball':
            x = F.dropout(x, self.dropout, training=self.training)

        if self.model_type == 'acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(
                        F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat(
                        [x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea = (self.gcns[0](x, adj_low, adj_high))

        if self.model_type == 'gcn' or self.model_type == 'mlp' or self.model_type == 'acmgcn':
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high)
        # return fea
        return F.log_softmax(fea, dim=1)


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


class WeightedRGATConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable

        super(WeightedRGATConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.use_edge_weights = True

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self.atten = Parameter(torch.Tensor(num_relations, 2*out_channels))
        self._alpha = None

        self.weight = Parameter(
            torch.Tensor(num_relations, in_channels[0], out_channels))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)
        glorot(self.atten)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_weight: OptTensor = None, edge_type: OptTensor = None):

        num_nodes = x.shape[0]

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight

        # No regularization/Basis-decomposition ========================
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            masked_edge_weight = None if edge_weight is None else edge_weight[edge_type == i]
            if x_l.dtype == torch.long:
                out += self.propagate(tmp, x=weight[i, x_l],
                                      size=size, masked_edge_weight=masked_edge_weight)
            else:
                x_tmp = x_l @ weight[i]
                h = self.propagate(tmp, x=x_tmp, size=size,
                                   atten=self.atten[i], masked_edge_weight=masked_edge_weight, rel=i, num_nodes=num_nodes)
                # out = out + (h @ weight[i])
                out = out + h

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, rel, num_nodes, x_i, x_j, index, masked_edge_weight, atten):

        x_cat = torch.cat((x_i, x_j), dim=1)
        alpha = (atten * x_cat).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, num_nodes=num_nodes)
        x_j = alpha.view(-1, 1) * x_j

        if self.use_edge_weights:
            return x_j if masked_edge_weight is None else masked_edge_weight.view(-1, 1) * x_j
        else:
            return x_j

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)

# extra_info
class WRGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, dims=16, drop=0, root=True):
        super(WRGAT, self).__init__()
        self.conv1 = WeightedRGATConv(
            num_features, dims, num_relations=num_relations, root_weight=root)
        self.conv2 = WeightedRGATConv(
            dims, num_classes, num_relations=num_relations, root_weight=root)

        self.drop = torch.nn.Dropout(p=drop)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_weight, edge_color):

        x = F.relu(self.conv1(x, edge_index,
                              edge_weight=edge_weight, edge_type=edge_color))
        x = self.drop(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight,
                       edge_type=edge_color)

        # return x
        return F.log_softmax(x, dim=1)