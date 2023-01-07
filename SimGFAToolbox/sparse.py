import torch
from torch_geometric.utils import sort_edge_index
import sklearn.preprocessing as pp
import warnings
warnings.filterwarnings('ignore')


def cosine_similarity_sparse(mat):
    """
    :param mat:
    :return:
    """
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat


def node_similarity_sparse(x):
    """
    :param x:
    :return:
    """
    sim = cosine_similarity_sparse(x)
    num_part = 400
    num_nodes = x.shape[0]
    parts = [[j for j in range(i, i + num_part) if j < num_nodes] for i in range(0, num_nodes, num_part)]
    print('part done')

    sim_sum = 0
    sim_num = 0
    all_sim = []
    for k, part_idx in enumerate(parts):
        new_matrix = []
        for m in part_idx:
            new_matrix.append(torch.Tensor(sim.getrow(m).toarray()))
        part_matrix = torch.concat(new_matrix, 0)
        all_sim.append(part_matrix)
        sim_sum += torch.sum(part_matrix)
        sim_num += len(part_idx) * num_nodes
    all_sim = torch.concat(all_sim, 0)
    sim_mean = sim_sum / sim_num
    print('Avg Node Similarity: {:.7f}'.format(sim_mean))
    return all_sim.reshape(-1, 1), sim_mean


def linked_node_similarity_sparse(x, edge_index):
    """
    :param x:
    :param edge_index:
    :return:
    """
    sim = cosine_similarity_sparse(x)
    num_nodes = x.shape[0]
    linked_nodes = []
    global_k = 0
    for i in range(num_nodes):
        linked_node = []
        for j in range(global_k, len(edge_index[0])):
            if edge_index[0][j] == i:
                linked_node.append(edge_index[1][j])
            else:
                global_k = j
                break
        linked_nodes.append(linked_node)
    print('part done')
    sim_sum = 0
    sim_num = 0
    all_sim = []
    for k in range(num_nodes):
        linked_node = torch.Tensor(linked_nodes[k]).to(torch.long)
        linked_sim = torch.Tensor(sim.getrow(k).toarray()).reshape(-1)[linked_node]
        all_sim.append(linked_sim)
        sim_sum += torch.sum(linked_sim)
        sim_num += len(linked_node)
    sim_mean = sim_sum / sim_num
    print('Avg Linked Node Similarity: {:.7f}'.format(sim_mean))
    all_sim = torch.concat(all_sim, -1)
    return all_sim.reshape(-1, 1), torch.mean(all_sim.reshape(-1, 1))


def neighborhood_similarity_sparse(x, edge_index):
    """
    :param x:
    :param edge_index:
    :return:
    """
    edge_index = sort_edge_index(edge_index)
    sim = cosine_similarity_sparse(x)
    num_nodes = x.shape[0]
    linked_nodes = []
    global_k = 0
    for i in range(num_nodes):
        linked_node = []
        for j in range(global_k, len(edge_index[0])):
            if edge_index[0][j] == i:
                linked_node.append(edge_index[1][j])
            else:
                global_k = j
                break
        linked_nodes.append(linked_node)
    print('part done')

    sim_sum = 0
    sim_num = num_nodes
    all_sim = []
    sim_mean_tmp = 0
    for k in range(num_nodes):
        linked_node = torch.Tensor(linked_nodes[k]).to(torch.long)
        if linked_node.shape[0] != 0:
            linked_sim = torch.Tensor(sim.getrow(k).toarray()).reshape(-1)[linked_node]
        else:
            linked_sim = torch.Tensor(0)
            linked_node = torch.Tensor(1)
            sim_num -= 1
        sim_avg = torch.sum(linked_sim) / linked_node.shape[0]
        sim_sum += sim_avg
        sim_mean_tmp += sim_avg * (1 / num_nodes)
        all_sim.append(sim_avg)
    print('Avg Neighborhood Similarity: {:.7f}'.format(sim_mean_tmp))
    all_sim = torch.Tensor(all_sim)
    return all_sim.reshape(-1, 1), sim_mean_tmp


def class_similarity_sparse(x, y):
    sim = cosine_similarity_sparse(x)
    n_classes = len(torch.unique(y))
    num_part = 400
    num_nodes = x.shape[0]
    parts = [[i for i in range(x, x + num_part) if i < num_nodes] for x in range(0, num_nodes, num_part)]
    print('part done')

    sim_matrix = torch.zeros(n_classes, n_classes)
    num_matrix = torch.zeros(n_classes, n_classes)
    for k, part_idx in enumerate(parts):
        if k % 100 == 0:
            print('part {}'.format(k))
        new_matrix = []
        new_y = y[part_idx]
        for m in part_idx:
            new_matrix.append(torch.Tensor(sim.getrow(m).toarray()))
        part_matrix = torch.concat(new_matrix, 0)
        for i in range(n_classes):
            for j in range(n_classes):
                index_i = torch.where(new_y == i)[0]
                index_j = torch.where(y == j)[0]
                tmp_sim = part_matrix[index_i, :]
                sim_index = tmp_sim[:, index_j]
                sim_ij = torch.sum(sim_index)
                sim_matrix[i, j] += sim_ij
                num_matrix[i, j] += len(index_i) * len(index_j)
    class_matrix = torch.div(sim_matrix, num_matrix)
    print('class matrix:', class_matrix)
    return class_matrix
