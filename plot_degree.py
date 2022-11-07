import time
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch.nn.functional as F
from torch_geometric.utils import degree, remove_self_loops, is_undirected, to_undirected
from torch_scatter import scatter_add, scatter, scatter_max, scatter_mean
from matplotlib.backends.backend_pdf import PdfPages
sns.set()
import warnings
warnings.filterwarnings('ignore')

from datasets import SNGNNPlanetoid, SNGNNWebKB, SNGNNActor, SNGNNWikipediaNetwork


def main():
    datasets = ['Texas', 'Cora', 'CiteSeer', 'PubMed',
                'Cornell', 'Wisconsin', 'Actor', 'Squirrel', 'Chameleon']
    # datasets = ['Cora']
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

        # if not is_undirected(dataset[0].edge_index):
        #     edge_index = to_undirected(dataset[0].edge_index)
        #     dataset_degree = degree(edge_index[0], dataset[0].num_nodes).tolist()
        # else:
        #     dataset_degree = degree(dataset[0].edge_index[0], dataset[0].num_nodes).tolist()

        dataset_degree = degree(dataset[0].edge_index[0], dataset[0].num_nodes).tolist()
        sum_degree = sum(dataset_degree)
        average_degree = sum_degree / dataset[0].num_nodes
        print("average degree of ", cfg['dataset'], " is ", average_degree)

        fig = plt.figure(figsize=(5, 4))
        sns.set_style('white')
        colors = ["teal", "amber", "greyish", "faded green", "dusty purple"]
        # denim blue  windows blue cyan darkcyan teal aqua blue midnightblue
        sns.xkcd_palette(colors)
        sns.distplot(dataset_degree, bins=200, hist=True, kde=False, rug=False, fit=None, hist_kws=None, kde_kws=None,
                     rug_kws=None, fit_kws=None, color=colors[0], vertical=False, norm_hist=False,
                     axlabel=None, label=None)
        plt.title('{} Degree Distribution\n(AvD: {:.3f})'.format(cfg['dataset'], average_degree),
                  fontsize=10, fontfamily='serif')
        plt.tick_params(direction='out', bottom=True, left=True, axis='both', width=1, length=4, colors='black')
        plt.xlabel(xlabel='degree', fontsize=10, fontfamily='serif')
        plt.figure(dpi=300)
        fig.savefig('./plot/degree/{}-degree.svg'.format(cfg['dataset']), bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()


