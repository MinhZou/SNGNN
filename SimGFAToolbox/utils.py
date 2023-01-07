import numpy as np
from scipy import sparse as sp


def edge_index_to_sparse_csc_tensor(x, edge_index):
    num_node = len(x)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.full(len(edge_index[0]), 1)
    matrix = sp.csc_matrix((data, (row, col)), shape=(num_node, num_node))
    return matrix