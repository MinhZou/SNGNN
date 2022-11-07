import torch
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize
from typing import Optional
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def edge_index_to_adj_mx(x, edge_index):
    edge_index = edge_index.to(torch.device('cpu'))
    x = x.to(torch.device('cpu'))
    num_node = len(x)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.full(len(edge_index[0]), 1)
    # mtrix = coo_matrix((data, (row, col)), shape=(num_node, num_node)).toarray()
    # return torch.Tensor(mtrix).to(torch.device('cuda'))
    mtrix = sp.coo_matrix((data, (row, col)), shape=(num_node, num_node))
    # return sparse_mx_to_torch_sparse_tensor(mtrix).to(torch.device('cuda'))
    return sparse_mx_to_torch_sparse_tensor(mtrix).to_dense()


def dense_to_sparse_coo_tensor(matrix):
    matrix = np.array(matrix)
    coo_np = sp.coo_matrix(matrix)
    return coo_np.tocsr()


def row_normalize(matrix):
    """Row-normalize sparse matrix"""
    # print(matrix)
    rowsum = np.array(matrix.sum(1))
    r_inv = np.power(rowsum, -1).flatten()

    # col_sum = np.array(matrix.sum(1))
    # r_inv = np.power(col_sum, -1).flatten()
    # print(r_inv.shape) # [N, 1]
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) # [N, N]
    # print(r_mat_inv)
    matrix = r_mat_inv.dot(matrix)
    # print(matrix.shape) # [N,N]
    return matrix


def edge_index_to_torch_coo_tensor(x, edge_index):
    adj = edge_index_to_adj_mx(x, edge_index.to(torch.device('cpu')))
    adj = dense_to_sparse_coo_tensor(adj)
    adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj = adj.to(torch.device('cuda'))
    # adj = adj.to(torch.device('cpu'))
    return adj


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)


def cosine_similarity(x):
    x = x / torch.norm(x, dim=-1, keepdim=True)
    similarity = torch.mm(x, x.T)
    return similarity


def edge_similarity_weight(x, edge_index):
    edge_similarity_weight = cosine_similarity(x)[edge_index[0].long(), edge_index[1].long()]
    return edge_similarity_weight


# test example
if __name__ == '__main__':
    # a = torch.tensor([[0, 1.2, 0], [2, 3.1, 0], [0.5, 0, 0]])
    # print(dense_to_sparse_coo_tensor(a))
    src = torch.Tensor([-1.0, 0.5, 1, 0.5, 0])
    index = torch.LongTensor([0, 0, 0, 1, 1])
    print(MinMaxScaler(src, index))
    print(RegionScaler(src, index))
