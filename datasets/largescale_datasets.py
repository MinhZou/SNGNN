import numpy as np
import torch
import scipy
import scipy.io
import pandas as pd
from sklearn.preprocessing import label_binarize
import gdown
from os import path
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import NodePropPredDataset
import os.path as osp
from torch_geometric.data import Dataset, download_url, Data

from utils.load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
from utils.data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url


class SNGNNGenius(Dataset):
    url = 'https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data/genius.mat'
    spilt_url = 'https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data/splits/genius-splits.npy'

    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)
        download_url(self.spilt_url, self.raw_dir)

    def get_idx_split_random(self, label, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        # if split_type == 'random':
        #     ignore_negative = False if self.name == 'ogbn-proteins' else True
        train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=train_prop,
                                                             valid_prop=valid_prop, ignore_negative=False)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        return split_idx

    def get_idx_split_fixed(self):
        data_path = osp.join(self.root, 'raw', 'genius-splits.npy')
        splits_lst = np.load(data_path, allow_pickle=True)
        print(len(splits_lst))
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        return splits_lst

    def process(self):
        data_path = osp.join(self.root, 'raw', 'genius.mat')
        fulldata = scipy.io.loadmat(data_path)
        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
        label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
        # print(label)
        split_idx = self.get_idx_split_random(label)
        # print(self.get_idx_split_fixed())
        # print(split_idx)
        train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
        data = Data(x=node_feat, edge_index=edge_index, y=label, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        # torch.save(self.collate([data]), self.processed_paths[0])
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data = self[0]
        y = data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return data.y.size(-1)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # if self.len() == 1:
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data


class SNGNNPenn94(Dataset):
    url = 'https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data/facebook100/Penn94.mat'
    spilt_url = 'https://github.com/CUAI/Non-Homophily-Large-Scale/tree/master/data/splits//fb100-Penn94-splits.npy'

    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)
        download_url(self.spilt_url, self.raw_dir)

    def process(self):
        from scipy.io import loadmat
        data_path = osp.join(self.root, 'raw', 'Penn94.mat')
        mat = loadmat(data_path)
        A = mat['A'].tocsr().tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        metadata = torch.from_numpy(mat['local_info'].astype('int64'))

        xs = []
        y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
        for i in range(x.size(1)):
            _, out = x[:, i].unique(return_inverse=True)
            xs.append(F.one_hot(out).to(torch.float))
        x = torch.cat(xs, dim=-1)

        data = Data(x=x, edge_index=edge_index, y=y)

        split_path = osp.join(self.root, 'raw', 'fb100-Penn94-splits.npy')
        splits = np.load(split_path, allow_pickle=True)
        # sizes = (data.num_nodes, len(splits))
        sizes = data.num_nodes
        train_masks, val_masks, test_masks = [], [], []
        # data.train_mask = torch.zeros(sizes, dtype=torch.bool)
        # data.val_mask = torch.zeros(sizes, dtype=torch.bool)
        # data.test_mask = torch.zeros(sizes, dtype=torch.bool)

        for i, split in enumerate(splits):
            tmp_train_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_val_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_test_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_train_mask[torch.tensor(split['train'])] = True
            tmp_val_mask[torch.tensor(split['valid'])] = True
            tmp_test_mask[torch.tensor(split['test'])] = True
            train_masks += [tmp_train_mask]
            val_masks += [tmp_val_mask]
            test_masks += [tmp_test_mask]
            # data.train_mask[:, i][torch.tensor(split['train'])] = True
            # data.val_mask[:, i][torch.tensor(split['valid'])] = True
            # data.test_mask[:, i][torch.tensor(split['test'])] = True
        data.train_mask = torch.stack(train_masks, dim=0)
        data.val_mask = torch.stack(val_masks, dim=0)
        data.test_mask = torch.stack(test_masks, dim=0)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data = self[0]
        y = data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return data.y.size(-1)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # if self.len() == 1:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data


class SNGNNSnapPatents(Dataset):
    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def get_idx_split_random(self, label, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        # if split_type == 'random':
        #     ignore_negative = False if self.name == 'ogbn-proteins' else True
        train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=train_prop,
                                                             valid_prop=valid_prop, ignore_negative=False)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        return split_idx

    def get_idx_split_fixed(self):
        data_path = osp.join(self.root, 'raw', 'snap-patents-splits.npy')
        splits_lst = np.load(data_path, allow_pickle=True)
        print(len(splits_lst))
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        return splits_lst

    def process(self):
        data_path = osp.join(self.raw_dir, 'snap-patents.mat')
        if not path.exists(data_path):
            p = '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia'
            print(f"Snap patents url: {p}")
            gdown.download(id=p, output=data_path, quiet=False)

        fulldata = scipy.io.loadmat(data_path)

        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(
            fulldata['node_feat'].todense(), dtype=torch.float)
        # num_nodes = int(fulldata['num_nodes'])

        years = fulldata['years'].flatten()
        nclass = 5
        label = even_quantile_labels(years, nclass, verbose=False)
        label = torch.tensor(label, dtype=torch.long).squeeze()
        data = Data(x=node_feat, edge_index=edge_index, y=label)
        # split_idx = self.get_idx_split_random(label)
        split_path = osp.join(self.root, 'raw', 'snap-patents-splits.npy')
        splits = np.load(split_path, allow_pickle=True)
        # sizes = (data.num_nodes, len(splits))
        sizes = data.num_nodes
        train_masks, val_masks, test_masks = [], [], []
        # data.train_mask = torch.zeros(sizes, dtype=torch.bool)
        # data.val_mask = torch.zeros(sizes, dtype=torch.bool)
        # data.test_mask = torch.zeros(sizes, dtype=torch.bool)

        for i, split in enumerate(splits):
            tmp_train_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_val_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_test_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_train_mask[torch.tensor(split['train'])] = True
            tmp_val_mask[torch.tensor(split['valid'])] = True
            tmp_test_mask[torch.tensor(split['test'])] = True
            train_masks += [tmp_train_mask]
            val_masks += [tmp_val_mask]
            test_masks += [tmp_test_mask]
            # data.train_mask[:, i][torch.tensor(split['train'])] = True
            # data.val_mask[:, i][torch.tensor(split['valid'])] = True
            # data.test_mask[:, i][torch.tensor(split['test'])] = True
        data.train_mask = torch.stack(train_masks, dim=0)
        data.val_mask = torch.stack(val_masks, dim=0)
        data.test_mask = torch.stack(test_masks, dim=0)
        # print(self.get_idx_split_fixed())
        # print(split_idx)
        # train_mask, val_mask, test_mask = split_idx['train'], split_idx['valid'], split_idx['test']
        # # data = Data(x=node_feat, edge_index=edge_index, y=label)
        # data = Data(x=node_feat, edge_index=edge_index, y=label, train_mask=train_mask,
        #            val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data = self[0]
        y = data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return data.y.size(-1)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # if self.len() == 1:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data


class SNGNNArxivYear(Dataset):
    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def get_idx_split_random(self, label, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        # if split_type == 'random':
        #     ignore_negative = False if self.name == 'ogbn-proteins' else True
        train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=train_prop,
                                                             valid_prop=valid_prop, ignore_negative=False)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        return split_idx

    def process(self):
        ogb_dataset = NodePropPredDataset(root='./datasets/data/raw_ogbn_arxiv', name='ogbn-arxiv')
        edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
        node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])

        nclass = 5
        label = even_quantile_labels(
            ogb_dataset.graph['node_year'].flatten(), nclass, verbose=False)
        label = torch.as_tensor(label).reshape(-1, 1).squeeze()
        data = Data(x=node_feat, edge_index=edge_index, y=label)

        split_path = osp.join(self.root, 'raw', 'arxiv-year-splits.npy')
        splits = np.load(split_path, allow_pickle=True)
        sizes = data.num_nodes
        train_masks, val_masks, test_masks = [], [], []

        for i, split in enumerate(splits):
            tmp_train_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_val_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_test_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_train_mask[torch.tensor(split['train'])] = True
            tmp_val_mask[torch.tensor(split['valid'])] = True
            tmp_test_mask[torch.tensor(split['test'])] = True
            train_masks += [tmp_train_mask]
            val_masks += [tmp_val_mask]
            test_masks += [tmp_test_mask]
        data.train_mask = torch.stack(train_masks, dim=0)
        data.val_mask = torch.stack(val_masks, dim=0)
        data.test_mask = torch.stack(test_masks, dim=0)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

        # ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
        # edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
        # node_feat= torch.as_tensor(ogb_dataset.graph['node_feat'])
        #
        # def ogb_idx_to_tensor(**kwargs):
        #     split_idx = ogb_dataset.get_idx_split()
        #     tensor_split_idx = {key: torch.as_tensor(
        #         split_idx[key]) for key in split_idx}
        #     return tensor_split_idx
        # train_masks, val_masks, test_masks = [], [], []
        #
        # for i, split in enumerate(splits):
        #     tmp_train_mask = torch.zeros(sizes, dtype=torch.bool)
        #     tmp_val_mask = torch.zeros(sizes, dtype=torch.bool)
        #     tmp_test_mask = torch.zeros(sizes, dtype=torch.bool)
        #     tmp_train_mask[torch.tensor(split['train'])] = True
        #     tmp_val_mask[torch.tensor(split['valid'])] = True
        #     tmp_test_mask[torch.tensor(split['test'])] = True
        #     train_masks += [tmp_train_mask]
        #     val_masks += [tmp_val_mask]
        #     test_masks += [tmp_test_mask]
        # data.train_mask = torch.stack(train_masks, dim=0)
        # data.val_mask = torch.stack(val_masks, dim=0)
        # data.test_mask = torch.stack(test_masks, dim=0)
        # ogb_dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
        # label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data = self[0]
        y = data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return data.y.size(-1)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # if self.len() == 1:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data


class SNGNNPokec(Dataset):
    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def get_idx_split_random(self, label, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        # if split_type == 'random':
        #     ignore_negative = False if self.name == 'ogbn-proteins' else True
        train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=train_prop,
                                                             valid_prop=valid_prop, ignore_negative=False)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        return split_idx

    def process(self):
        data_path = osp.join(self.raw_dir, 'pokec.mat')
        if not path.exists(data_path):
            gdown.download(id=dataset_drive_url['pokec'], \
                           output=data_path, quiet=False)

        fulldata = scipy.io.loadmat(data_path)

        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat']).float()
        num_nodes = int(fulldata['num_nodes'])
        normalize = True
        if normalize:
            node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
            node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

        label = fulldata['label'].flatten()
        label = torch.tensor(label, dtype=torch.long)
        data = Data(x=node_feat, edge_index=edge_index, y=label)

        split_path = osp.join(self.root, 'raw', 'pokec-splits.npy')
        splits = np.load(split_path, allow_pickle=True)
        sizes = data.num_nodes
        train_masks, val_masks, test_masks = [], [], []

        for i, split in enumerate(splits):
            tmp_train_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_val_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_test_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_train_mask[torch.tensor(split['train'])] = True
            tmp_val_mask[torch.tensor(split['valid'])] = True
            tmp_test_mask[torch.tensor(split['test'])] = True
            train_masks += [tmp_train_mask]
            val_masks += [tmp_val_mask]
            test_masks += [tmp_test_mask]
        data.train_mask = torch.stack(train_masks, dim=0)
        data.val_mask = torch.stack(val_masks, dim=0)
        data.test_mask = torch.stack(test_masks, dim=0)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data = self[0]
        y = data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return data.y.size(-1)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # if self.len() == 1:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data


class SNGNNTwitchGamer(Dataset):
    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def get_idx_split_random(self, label, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        # if split_type == 'random':
        #     ignore_negative = False if self.name == 'ogbn-proteins' else True
        train_idx, valid_idx, test_idx = rand_train_test_idx(label, train_prop=train_prop,
                                                             valid_prop=valid_prop, ignore_negative=False)
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
        return split_idx

    def process(self):
        feat_data_path = osp.join(self.raw_dir, 'twitch-gamer_feat.csv')
        edge_data_path = osp.join(self.raw_dir, 'twitch-gamer_edges.csv')
        if not path.exists(feat_data_path):
            gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
                           output=feat_data_path, quiet=False)
        if not path.exists(edge_data_path):
            gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
                           output=edge_data_path, quiet=False)

        edges = pd.read_csv(edge_data_path)
        nodes = pd.read_csv(feat_data_path)
        edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
        # num_nodes = len(nodes)
        label, features = load_twitch_gamer(nodes, task="mature")
        node_feat = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        # label = torch.tensor(label)
        normalize = True
        if normalize:
            node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
            node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

        data = Data(x=node_feat, edge_index=edge_index, y=label)

        split_path = osp.join(self.root, 'raw', 'twitch-gamer-splits.npy')
        splits = np.load(split_path, allow_pickle=True)
        sizes = data.num_nodes
        train_masks, val_masks, test_masks = [], [], []

        for i, split in enumerate(splits):
            tmp_train_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_val_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_test_mask = torch.zeros(sizes, dtype=torch.bool)
            tmp_train_mask[torch.tensor(split['train'])] = True
            tmp_val_mask[torch.tensor(split['valid'])] = True
            tmp_test_mask[torch.tensor(split['test'])] = True
            train_masks += [tmp_train_mask]
            val_masks += [tmp_val_mask]
            test_masks += [tmp_test_mask]
        data.train_mask = torch.stack(train_masks, dim=0)
        data.val_mask = torch.stack(val_masks, dim=0)
        data.test_mask = torch.stack(test_masks, dim=0)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data = self[0]
        y = data.y
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(data.y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            return torch.unique(y).numel()
        else:
            return data.y.size(-1)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # if self.len() == 1:
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.load(osp.join(self.processed_dir, f'data.pt'))
        return data

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'twitch-e':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(sub_dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset()
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif dataname == 'arxiv-year':
        dataset = load_arxiv_year_dataset()
    elif dataname == 'pokec':
        dataset = load_pokec_mat()
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat()
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset()
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(dataname)
    elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
    elif dataname == "twitch-gamer":
        dataset = load_twitch_gamer_dataset()
    elif dataname == "wiki":
        dataset = load_wiki()
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset


def load_proteins_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins')
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}

    dataset.get_idx_split = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    return dataset


def load_ogb_dataset(name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name)
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
                       output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
                       output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_yelpchi_dataset():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
                       output=f'{DATAPATH}YelpChi.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_planetoid_dataset(name):
    torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    def planetoid_orig_split(**kwargs):
        return {'train': torch.as_tensor(dataset.train_idx),
                'valid': torch.as_tensor(dataset.valid_idx),
                'test': torch.as_tensor(dataset.test_idx)}

    dataset.get_idx_split = planetoid_orig_split
    dataset.label = label

    return dataset


def load_geom_gcn_dataset(name):
    fulldata = scipy.io.loadmat(f'{DATAPATH}/{name}.mat')
    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
                       output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
                       output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)

    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():
    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
                       output=f'{DATAPATH}wiki_features2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
                       output=f'{DATAPATH}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
                       output=f'{DATAPATH}wiki_views2M.pt', quiet=False)

    dataset = NCDataset("wiki")
    features = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edges = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt')
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges,
                     "edge_feat": None,
                     "node_feat": features,
                     "num_nodes": num_nodes}
    dataset.label = label
    return dataset
