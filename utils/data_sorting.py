import torch
import numpy as np


def reorder_data(features, edge_index, labels):
    """ reorder the node index in a map.
    :param features: features
    :param edge_index: edge (node pair)
    :param labels: labels
    :return:
    """
    # Indices shape the same as features
    _, indices = torch.sort(features, descending=False, stable=True, dim=0)
    indices_order = torch.unsqueeze(indices[:, 0], dim=1)
    map_dic = {}
    for i in range(len(indices)):
        map_dic[int(indices[:, 0][i])] = i

    def map_index(x, *y):
        return map_dic.get(x)

    # map_ is only implemented on CPU tensors
    edge_index.map_(edge_index, map_index)
    indices_features = indices_order.expand_as(indices)
    indices_labels = indices[:, 0]
    features = torch.gather(features, 0, indices_features)
    labels = torch.gather(labels, -1, indices_labels)
    return features, edge_index, labels


def lexsort_torch(features, edge_index, labels):
    """
    sort the features using numpy.lexsort() function
    :param features: shape [num_node, features_dim]
    :param edge_index: shape [num_edge, 2] or [2, num_edge]
    :param labels: shape [num_node, 1]
    :return: sorted features, edge_index, labels
    """
    data = features.numpy()
    data_index = [data[:, i] for i in range(len(data[0]))][::-1]
    index = np.lexsort(data_index)

    map_dic = {}
    for i in range(len(index)):
        map_dic[int(index[i])] = i

    def map_index(x, *y):
        return map_dic.get(x)

    edge_index.map_(edge_index, map_index)
    features, labels = features[index], labels[index]
    return features, edge_index, labels
