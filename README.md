# Similarity-Navigated Graph Neural Networks for Node Classification

### Introduction

This repository contains a pytorch implementation of **Similarity-Navigated Graph Neural Networks for Node Classification**

Graph Neural Networks are effective in learning representations of graph-structured data. Some recent works are devoted to addressing heterophily, which exists ubiquitously in real-world networks, breaking the homophily assumption that nodes belonging to the same class are more likely to be connected and restricting the generalization of traditional methods in tasks such as node classification. However, these heterophily-oriented methods still lose efficacy in some typical heterophilic datasets. Moreover, issues on leveraging the knowledge from both node features and graph structure and investigating inherent properties of the datasets still need further consideration. In this work, we first provide insights based on similarity metrics to interpret the long-existing confusion that simple models sometimes perform better than models dedicated to heterophilic networks. Then, sticking to these insights and the classification principle of narrowing the intra-class distance and enlarging the inter-class distance of the sample's embeddings, we propose a Similarity-Navigated Graph Neural Network (SNGNN) which uses *Node Similarity* matrix coupled with $\texttt{mean}$ aggregation operation instead of the normalized adjacency matrix in the neighborhood aggregation process. Moreover, based on SNGNN, a novel explicitly aggregating mechanism for selecting similar neighbors, named SNGNN+, is devised to preserve distinguishable features and handle the heterophilic problem. Additionally, a variant, SNGNN++, is further designed to adaptively integrate the knowledge from both node features and graph structure for improvement. Extensive experiments are conducted and demonstrate that our proposed framework outperforms the state-of-the-art methods for both small-scale and large-scale graphs regardless of their heterophilic extent. Our implementation is available online.

### Installation

The requirement.txt included all the dependencies. It main depends on:

- python=3.7.13
- torch=1.10.0+cu111
- torch-geometric=2.0.4
- torch-scatter=2.0.9
- torch-sparse=0.6.13
- torchvision=0.11.0+cu111
- scipy=1.7.3
- prettytable=3.3.0
- pyyaml=6.0
- icecream=2.1.2
- numpy=1.21.6
- pandas=1.3.5
- scikit-learn=1.0.2
- matplotlib=3.5.2
- seaborn=0.11.2
- tqdm=4.64.0

PyG is the abbreviation of torch geometric.  To install PyG, you should install torch in advance. If you use GPU version, you also need to install CUDA in advance. The relevant packages torch-scatter, torch-sparse is also needed. PyG is now available for Python 3.7 to Python 3.10.  More installation details refer to the official documentation: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#

CUDA and Torch official  installation reference:

https://developer.nvidia.cn/

https://pytorch.org/

An example of install PyG relevant packages: 

{TORCH}, {CUDA} is the installed torch and cuda version, respectively.

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

You can able to install other dependencies using pip (`pip install xxx`). If you do not yet have Python installed, I recommend starting with [Anaconda](https://www.anaconda.com/products/distribution). 



### Example usage

The dataset and splits will be downloaded and processed automatically for the first time.  Therefore, just download the repository and run it in the following ways after solving the environment.

To train and evaluate SNGNN_Plus_Plus for node classification with the command line:

```
python train.py --config ./config/config-test.yaml --data_splits --model SNGNN_Plus_Plus --num_layers 1 --dataset Chameleon --seed 3 --patience 200 --hidden_channels 32 --part_id 1 --top_k 10 --thr 0.9 --init_beta 0.0 --is_remove_self_loops 1 --dropout 0.5 --epochs 2000 --lr 0.01 --weight_decay 0.0005
```

Using the shell script to train with all settings:

```
sh train_script_SNGNN_Plus_Plus.sh
```



### Similarity-Navigated Graph Feature Analysis Toolbox (Sim-GFA Toolbox)

![Sim-GFA Toolbox](https://raw.githubusercontent.com/MinhZou/SNGNN/main/pic/Sim-GFA%20Toolbox.png)

<center> Figure 7: A framework illustration of Sim-GFA Toolbox. </center>

