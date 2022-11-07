import time
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch.nn.functional as F
from torch_geometric.utils import degree, remove_self_loops, is_undirected, to_undirected, to_dense_adj
from torch_scatter import scatter_add, scatter, scatter_max, scatter_mean
from matplotlib.backends.backend_pdf import PdfPages
sns.set()
import warnings
warnings.filterwarnings('ignore')

from datasets import SNGNNPlanetoid, SNGNNWebKB, SNGNNActor, SNGNNWikipediaNetwork


# def cosine_similarity(x):
#     x = x / torch.norm(x, dim=-1, keepdim=True)
#     similarity = torch.mm(x, x.T)
#     return similarity

def cosine_similarity(x):
    norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
    sim = norm.mm(norm.t())
    return sim


def node_similarity(x):
    sim = cosine_similarity(x)
    I = torch.eye(sim.shape[0], dtype=torch.uint8)
    mask = torch.ones_like(torch.Tensor(sim.shape[0]), dtype=torch.uint8) - I
    sim = sim[mask]
    # tmp_sim = (sim + I).reshape(-1, 1)
    # index, _ = torch.where(tmp_sim <= 1.0)
    # sim = torch.index_select(sim.reshape(-1, 1), 0, index).numpy()
    # print(torch.mean(sim))
    return sim


def link_node_similarity(x, edge_index):
    sim = cosine_similarity(x)
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index, None)
    sim = sim[edge_index[0], edge_index[1]]
    print(sim.shape)
    return sim.reshape(-1, 1)


def neighborhood_similarity(x, edge_index):
    norm = F.normalize(x, p=2., dim=-1)
    edge_index, _ = remove_self_loops(edge_index, None)
    sim_i = torch.index_select(norm, 0, edge_index[0])
    sim_j = torch.index_select(norm, 0, edge_index[1])
    sim = (sim_i * sim_j).sum(dim=-1)
    weight = scatter_mean(sim, edge_index[0], dim=0)
    return weight


def class_similarity(x, y, edge_index, cfg):
    # norm = F.normalize(x, p=2., dim=-1)
    sim = cosine_similarity(x)
    # print(sim)
    n_classes = len(torch.unique(y))
    edge_index, _ = remove_self_loops(edge_index, None)
    intra_class_similarity = []
    sim_matrix = torch.zeros(n_classes, n_classes)
    for i in range(n_classes):
        for j in range(n_classes):
            # print(len(y == i))
            index_i = torch.where(y == i)[0]
            index_j = torch.where(y == j)[0]
            tmp_sim = sim[index_i, :]
            sim_index = tmp_sim[:, index_j]
            # print(y.shape, sim_index.shape)
            sim_ij = torch.mean(sim_index)
            sim_matrix[i, j] = sim_ij
    # plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    fig, ax = plt.subplots()
    sns.heatmap(sim_matrix, mask=mask, annot=True, cmap='viridis')
    if cfg['use_adj'] or not cfg['use_features']:
        plt.title('{} ({})'.format('Class Similarity', cfg['dataset']),
                  fontsize=15, fontfamily='serif')
    else:
        plt.title('{}\n({}, {})'.format('Class Similarity', cfg['dataset'], cfg['model']),
                  fontsize=15, fontfamily='serif')
    plt.figure(dpi=300)
    # fig.savefig('./plot/{}-{}-{}.pdf'.format(cfg['dataset'], cfg['model'],  'Class Similarity'), format='pdf')
    if cfg['use_features']:
        fig.savefig('./plot/features/{}-{}-{}.svg'.format(cfg['dataset'], cfg['model'], 'Class Similarity'))
    elif cfg['use_adj']:
        fig.savefig('./plot/adj/{}-{}-{}.svg'.format(cfg['dataset'], cfg['model'], 'Class Similarity'))
    else:
        fig.savefig('./plot/similarity/{}-{}-{}.svg'.format(cfg['dataset'], cfg['model'], 'Class Similarity'))
    plt.show()
    plt.close()
    return sim_matrix


def plot_similarity(data, cfg):
    x, edge_index, y = data.x, data.edge_index, data.y
    if cfg['use_adj']:
        x = to_dense_adj(edge_index).squeeze(dim=0)
        # print(x.shape)
    sim = None
    if cfg['similarity'] == 'Node Similarity':
        sim = node_similarity(x)
    elif cfg['similarity'] == 'Linked Node Similarity':
        sim = link_node_similarity(x, edge_index)
    elif cfg['similarity'] == 'Neighborhood Similarity':
        sim = neighborhood_similarity(x, edge_index)
    elif cfg['similarity'] == 'Class Similarity':
        class_similarity(x, y, edge_index, cfg)
        return
    else:
        print('wrong setting for similiarty')
        return
    # sns.set_style('ticks')
    sns.set_style('white')
    colors = ["darkblue", "amber", "greyish", "faded green", "dusty purple"]
    # denim blue  windows blue cyan darkcyan teal aqua blue midnightblue
    sns.xkcd_palette(colors)
    # fig, ax = plt.subplots(figsize=(5, 4))
    fig = plt.figure(figsize=(5, 4))
    sns.distplot(sim, bins=200, hist=True, kde=False, rug=False, fit=None, hist_kws=None, kde_kws=None,
                 rug_kws=None, fit_kws=None, color=colors[0], vertical=False, norm_hist=False,
                 axlabel=None, label=None)
    # sns.distplot(sim, bins=200, hist=True, norm_hist=True, color=colors[0], ax=ax)
    # plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.set_title(title, fontsize=15, fontfamily='serif')
    if cfg['use_features']:
        plt.title('{} Frequency Distribution\n({}, {}, Avg: {:.3f})'.format(cfg['similarity'],
                                                                         cfg['dataset'], cfg['model'],
                                                                         torch.mean(sim)), fontsize=8,fontfamily='serif')
    elif cfg['use_adj']:
        plt.title('{} Frequency Distribution\n({}, Avg: {:.3f})'.format(cfg['similarity'],
                                                                     cfg['dataset'],
                                                                     torch.mean(sim)), fontsize=10, fontfamily='serif')
    else:
        plt.title('{} Frequency Distribution\n({}, Avg: {:.3f})'.format(cfg['similarity'],
                                                                        cfg['dataset'],
                                                                        torch.mean(sim)), fontsize=10,
                  fontfamily='serif')
    # ax.set_ylabel(ylabel='Count', fontsize=10, fontfamily='serif')
    # sns.despine(left=True, bottom=True)
    plt.tick_params(direction='out', bottom=True, left=True, axis='both', width=1, length=4, colors='black')
    # ax.set_xticks([-1, -0.5, 0, 0.5, 1], size=10)
    plt.xlabel(xlabel='Similarity', fontsize=8, fontfamily='serif')
    # plt.xlim([-1, 1])
    # plt.legend()
    plt.figure(dpi=300)
    # fig = dis_fig.get_figure()
    if cfg['use_features']:
        fig.savefig('./plot/features/{}-{}-{}.svg'.format(cfg['dataset'], cfg['model'], cfg['similarity']),
                    bbox_inches='tight')
    elif cfg['use_adj']:
        fig.savefig('./plot/adj/{}-{}-{}.svg'.format(cfg['dataset'], cfg['model'], cfg['similarity']),
                    bbox_inches='tight')
    else:
        fig.savefig('./plot/similarity/{}-{}-{}.svg'.format(cfg['dataset'], cfg['model'], cfg['similarity']),
                    bbox_inches='tight')

    plt.show()
    plt.close()
    return

def main():
    # datasets = ['Texas', 'Cora', 'CiteSeer', 'PubMed',
    #             'Cornell', 'Wisconsin', 'Actor', 'Squirrel', 'Chameleon']
    datasets = ['Wisconsin']
    for data_name in datasets:
        print(data_name)
        cfg = {}
        cfg['dataset'] = data_name
        # dataset setting
        if cfg['dataset'] == 'Cora':
            dataset = SNGNNPlanetoid(
                root='./datasets/data/Cora', name='Cora')
        elif cfg['dataset'] == 'CiteSeer':
            dataset = SNGNNPlanetoid(
                root='./datasets/data/CiteSeer', name='CiteSeer')
        elif cfg['dataset'] == 'PubMed':
            dataset = SNGNNPlanetoid(
                root='./datasets/data/PubMed', name='PubMed')
        elif cfg['dataset'] == 'Cornell':
            dataset = SNGNNWebKB(
                root='./datasets/data/Cornell', name='Cornell')
            dataset.process()
        elif cfg['dataset'] == 'Texas':
            dataset = SNGNNWebKB(
                root='./datasets/data/Texas', name='Texas')
            dataset.process()
        elif cfg['dataset'] == 'Wisconsin':
            dataset = SNGNNWebKB(
                root='./datasets/data/Wisconsin', name='Wisconsin')
            dataset.process()
        elif cfg['dataset'] == 'Actor':
            dataset = SNGNNActor(
                root='./datasets/data/Actor')
            dataset.process()
        elif cfg['dataset'] == 'Chameleon':
            dataset = SNGNNWikipediaNetwork(
                root='./datasets/data/chameleon', name='chameleon')
            dataset.process()
        elif cfg['dataset'] == 'Squirrel':
            dataset = SNGNNWikipediaNetwork(
                root='./datasets/data/squirrel', name='squirrel')
            dataset.process()
        else:
            return

        data = dataset[0]
        use_features = False
        use_adj = False
        cfg['use_features'] = use_features
        cfg['use_adj'] = use_adj
        similarity = ['Node Similarity', 'Linked Node Similarity', 'Neighborhood Similarity', 'Class Similarity']
        # similarity = ['Class Similarity']
        # models = ['GAT', 'GCN', 'MLP', 'LINK','SNGNN', 'SNGNN_Plus', 'SNGNN_Plus_Plus']
        models = ['SNGNN']
        for s in similarity:
            cfg['similarity'] = s
            if not use_features:
                cfg['model'] = ''
                plot_similarity(data, cfg)
            else:
                for m in models:
                    cfg['model'] = m  
                    save_dir = './features'
                    for file in os.listdir(save_dir):
                        info = file.split('_')
                        if cfg['model'] == info[0] and cfg['dataset'] == info[1]:
                            print(file)
                            path = os.path.join(save_dir, file)
                            data.x = torch.load(path).cpu()
                            print(cfg['model'], cfg['dataset'], cfg['similarity'])
                            plot_similarity(data, cfg)
                        else:
                            pass


if __name__ == '__main__':
    main()


