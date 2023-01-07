import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()


def plot_class_similarity(class_matrix, dataset_name='', model_name='', graph=False, learned_embeddings=False):
    """
    :param class_matrix: n x n matrix, n denotes the number of class
    :param dataset_name: str
    :param model_name: str
    :param graph: bool, whether use adjacency matrix as input
    :param learned_embeddings: bool, whether use learned_embeddings as input
    :return:
    """
    mask = np.triu(np.ones_like(class_matrix, dtype=bool), k=1)
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 10})
    sns.heatmap(class_matrix, mask=mask, annot=True, cmap='YlGnBu_r', fmt='.4g')  # YlGnBu_r viridis
    plt.title('{}'.format('Class Similarity'), fontsize=30, fontfamily='serif')
    plt.figure(dpi=300)
    if learned_embeddings:
        save_dir = './plot/learned_embeddings/' + 'class_similarity'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + '/{}-{}-{}-embeddings.svg'.format(dataset_name, model_name, 'Class Similarity'))
    elif graph:
        save_dir = './plot/graph/' + 'class_similarity'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + '/{}-{}-{}-graph.svg'.format(dataset_name, model_name, 'Class Similarity'))
    else:
        save_dir = './plot/node/' + 'class_similarity'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + '/{}-{}-{}.svg'.format(dataset_name, model_name, 'Class Similarity'))
    plt.show()
    plt.close()
    return


def plot_similarity_distribution(sim, avg_sim,  similarity_type='', dataset_name='', model_name='',
                                 graph=False, learned_embeddings=False):
    """
    :param sim: shape(n, -1), n denote the number of nodes or edges
    :param avg_sim: float
    :param similarity_type: str
    :param dataset_name: str
    :param model_name: str
    :param graph: bool, whether use adjacency matrix as input
    :param learned_embeddings: bool, whether use learned_embeddings as input
    :return:
    """
    if sim is None:
        return
    sns.set_style('white')
    colors = ["darkblue"]
    sns.color_palette("Blues_r")
    fig = plt.figure(figsize=(5, 4))
    sns.distplot(sim, bins=200, hist=True, kde=False, rug=False, fit=None, hist_kws=None, kde_kws=None,
                 rug_kws=None, fit_kws=None, color=colors[0], vertical=False, norm_hist=False,
                 axlabel=None, label=None)
    plt.title('Avg: {:.7f}'.format(avg_sim), fontsize=15, fontfamily='serif')
    plt.tick_params(direction='out', bottom=True, left=True, axis='both', width=1, length=4, colors='black')
    plt.xlabel(xlabel='{}'.format(similarity_type), fontsize=15, fontfamily='serif')
    plt.figure(dpi=300)
    if learned_embeddings:
        save_dir = './plot/learned_embeddings/' + str(similarity_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir+'/{}-{}-{}-embeddings.svg'.format(dataset_name, model_name,
                                                               similarity_type), bbox_inches='tight')
    elif graph:
        save_dir = './plot/graph/' + str(similarity_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir+'/{}-{}-{}-graph.svg'.format(dataset_name, model_name,
                                                          similarity_type), bbox_inches='tight')
    else:
        save_dir = './plot/node/' + str(similarity_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir+'/{}-{}-{}-node.svg'.format(dataset_name, model_name,
                                                         similarity_type), bbox_inches='tight')
    plt.show()
    plt.close()
    return
