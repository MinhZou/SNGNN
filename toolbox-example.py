from torch_geometric.utils import remove_self_loops, to_dense_adj

from datasets import SNGNNPenn94
from SimGFAToolbox import linked_node_similarity_sparse, linked_node_similarity_dense_small, \
    linked_node_similarity_dense_large, plot_similarity_distribution, edge_index_to_sparse_csc_tensor

if __name__ == '__main__':
    dataset = SNGNNPenn94(root='./datasets/data/Penn94', name='Penn94')
    similarity = 'Linked Node Similarity'
    data = dataset[0]
    x, y, edge_index = data.x, data.y, data.edge_index
    edge_index = remove_self_loops(edge_index, None)
    edge_index = edge_index[0]

    # dense for small-scale datsets
    sim, avg_sim = linked_node_similarity_dense_small(x, edge_index)
    plot_similarity_distribution(sim, avg_sim, dataset_name='Penn94', similarity_type=similarity)
    # dense large for large-scale datasets
    sim, avg_sim = linked_node_similarity_dense_large(x, edge_index)
    plot_similarity_distribution(sim, avg_sim, dataset_name='Penn94', similarity_type=similarity)

    # dense for adjacency matrix
    x_d = to_dense_adj(edge_index).squeeze()
    sim, avg_sim = linked_node_similarity_dense_small(x_d, edge_index)
    plot_similarity_distribution(sim, avg_sim, dataset_name='Penn94', similarity_type=similarity, graph=True)

    # sparse for adjacency matrix
    x_s = edge_index_to_sparse_csc_tensor(x, edge_index)
    sim, avg_sim = linked_node_similarity_sparse(x_s, edge_index)
    plot_similarity_distribution(sim, avg_sim, dataset_name='Penn94', similarity_type=similarity, graph=True)
