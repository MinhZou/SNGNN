import torch
import torch.nn.functional as F
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter_mean
import warnings
warnings.filterwarnings('ignore')


def node_similarity_dense_large_parted(x):
    """
    For large-scale datasets. The version of parallel calculation is under development.
    :param x:
    :return:
    """
    norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
    num_nodes = norm.shape[0]
    num_part = 1000
    parts = [[j for j in range(i, i + num_part) if j < num_nodes] for i in range(0, num_nodes, num_part)]
    print('part done')

    sim_sum = 0
    for k, part_idx in enumerate(parts):
        new_matrix = []
        for part_idx2 in parts:
            new_matrix.append(torch.mm(norm[part_idx, :], norm.T[:, part_idx2]))
        part_matrix = torch.concat(new_matrix, -1)
        sim_sum += torch.sum(part_matrix)
    sim_mean = (sim_sum - num_nodes) / (num_nodes - 1)*num_nodes
    print('Avg Node Similarity: {:.7f}'.format(sim_mean))
    return None, sim_mean


def linked_node_similarity_dense_large(x, edge_index):
    edge_index = sort_edge_index(edge_index)
    norm = F.normalize(x, p=2., dim=-1)
    num_nodes = norm.shape[0]
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
        linked_sim = torch.mm(norm[k, :].reshape(1, -1), norm.T).reshape(-1)[linked_node]
        all_sim.append(linked_sim)
        sim_sum += torch.sum(linked_sim)
        sim_num += len(linked_node)
    sim_mean = sim_sum / sim_num
    print('Avg Linked Node Similarity: {:.6f}'.format(sim_mean))
    all_sim = torch.concat(all_sim, -1)
    return all_sim.reshape(-1, 1), torch.mean(all_sim.reshape(-1, 1))


def neighborhood_similarity_dense_large(x, edge_index):
    edge_index = sort_edge_index(edge_index)
    norm = F.normalize(x, p=2., dim=-1)
    num_nodes = norm.shape[0]
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
            linked_sim = torch.mm(norm[k, :].reshape(1, -1), norm.T).reshape(-1)[linked_node]
        else:
            linked_sim = torch.Tensor(0)
            linked_node = torch.Tensor(1)
            sim_num -= 1
        sim_avg = torch.sum(linked_sim) / linked_node.shape[0]
        sim_sum += sim_avg
        sim_mean_tmp += sim_avg*(1/num_nodes)
        all_sim.append(sim_avg)

    print('Avg Neighborhood Similarity: {:.7f}'.format(sim_mean_tmp))
    all_sim = torch.Tensor(all_sim)
    return all_sim.reshape(-1, 1), sim_mean_tmp


def class_similarity_dense_large(x, y):
    n_classes = len(torch.unique(y))
    norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
    num_nodes = norm.shape[0]
    num_part = 1000
    parts = [[j for j in range(i, i + num_part) if j < num_nodes] for i in range(0, num_nodes, num_part)]
    print('part done')

    sim_matrix = torch.zeros(n_classes, n_classes)
    num_matrix = torch.zeros(n_classes, n_classes)
    for k, part_idx in enumerate(parts):
        new_matrix = []
        new_y = y[part_idx]
        for part_idx2 in parts:
            new_matrix.append(torch.mm(norm[part_idx, :], norm.T[:, part_idx2]))
        part_matrix = torch.concat(new_matrix, -1)
        for i in range(n_classes):
            for j in range(n_classes):
                index_i = torch.where(new_y == i)[0]
                index_j = torch.where(y == j)[0]
                tmp_sim = part_matrix[index_i, :]
                sim_index = tmp_sim[:, index_j]
                sim_ij = torch.sum(sim_index)
                sim_matrix[i, j] += sim_ij
                num_matrix[i, j] += len(index_i)*len(index_j)
    class_matrix = torch.div(sim_matrix, num_matrix)
    return class_matrix


# def cosine_similarity_small(x):
#     x = x / torch.norm(x, dim=-1, keepdim=True)
#     similarity = torch.mm(x, x.T)
#     return similarity

def cosine_similarity_dense_small(x):
    norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
    sim = norm.mm(norm.t())
    return sim


def node_similarity_dense_small(x):
    sim = cosine_similarity_dense_small(x)
    I = torch.eye(sim.shape[0], dtype=torch.uint8)
    mask = torch.ones_like(torch.Tensor(sim.shape[0]), dtype=torch.uint8) - I
    sim = sim[mask]
    return sim, torch.mean(sim)


def linked_node_similarity_dense_small(x, edge_index):
    sim = cosine_similarity_dense_small(x)
    sim = sim[edge_index[0], edge_index[1]]
    return sim.reshape(-1, 1), torch.mean(sim)


def neighborhood_similarity_dense_small(x, edge_index):
    norm = F.normalize(x, p=2., dim=-1)
    sim_i = torch.index_select(norm, 0, edge_index[0])
    sim_j = torch.index_select(norm, 0, edge_index[1])
    sim = (sim_i * sim_j).sum(dim=-1)
    weight = scatter_mean(sim, edge_index[0], dim=0)
    return weight, torch.mean(weight)


def class_similarity_dense_small(x, y):
    sim = cosine_similarity_dense_small(x)
    n_classes = len(torch.unique(y))
    sim_matrix = torch.zeros(n_classes, n_classes)
    for i in range(n_classes):
        for j in range(n_classes):
            index_i = torch.where(y == i)[0]
            index_j = torch.where(y == j)[0]
            tmp_sim = sim[index_i, :]
            sim_index = tmp_sim[:, index_j]
            sim_ij = torch.mean(sim_index)
            sim_matrix[i, j] = sim_ij
    return sim_matrix, torch.mean(sim_matrix)





