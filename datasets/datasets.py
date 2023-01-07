import warnings
import sys

from torch_geometric.data.dataset import to_list, files_exist, makedirs,_repr
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
import torch
import os.path as osp
import numpy as np
from torch_sparse import SparseTensor, coalesce
from typing import Callable, List, Optional
from torch_geometric.data import Data, download_url, extract_tar
from torch_geometric.io import read_planetoid_data



class SNGNNPlanetoid(Planetoid):
    """docstring for SNGNNPlanetoid"""

    # url = 'https://gitee.com/jiajiewu/planetoid/raw/master/data'
    # url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    # url = 'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data'
    # splits_url = 'https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/splits'
    splits_url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/splits'

    def __init__(self, root: str, name: str, data_splits=True):
        self.data_splits = data_splits

        super().__init__(root, name)
        if self.data_splits:
            if 'download_splits' in self.__class__.__dict__:
                self._download_splits()
            if 'process_spilts' in self.__class__.__dict__:
                self._process_splits()
        else:
            if 'download' in self.__class__.__dict__:
                self._download()
            if 'process' in self.__class__.__dict__:
                self._process()

        if self.data_splits:
            self.data, self.slices = torch.load(self.processed_splits_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def raw_splits_names(self):
        return [f'{self.name.lower()}_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def raw_splits_dir(self) -> str:
        return osp.join(self.root, self.name, 'splits')

    def download_splits(self):
        for name in self.raw_splits_names:
            download_url(f'{self.splits_url}/{name}', self.raw_splits_dir)

    @property
    def processed_splits_dir(self):
        return osp.join(self.root, self.name, 'splits_processed')

    @property
    def raw_splits_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = to_list(self.raw_splits_names)
        return [osp.join(self.raw_splits_dir, f) for f in files]

    @property
    def processed_splits_file_names(self) -> str:
        return 'data.pt'

    @property
    def processed_splits_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = to_list(self.processed_splits_file_names)
        return [osp.join(self.processed_splits_dir, f) for f in files]

    def process_spilts(self):
        # data source, so self.raw_dir doesn't need to change
        data = read_planetoid_data(self.raw_dir, self.name)
        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_splits_paths[:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=0)
        val_mask = torch.stack(val_masks, dim=0)
        test_mask = torch.stack(test_masks, dim=0)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_splits_paths[0])

    def _download_splits(self):
        if files_exist(self.raw_splits_paths):  # pragma: no cover
            return

        makedirs(self.raw_splits_dir)
        self.download_splits()

    def _process_splits(self):
        f = osp.join(self.processed_splits_dir, 'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_splits_dir}' first")

        f = osp.join(self.processed_splits_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_splits_paths):  # pragma: no cover
            return

        print('Processing...', file=sys.stderr)

        makedirs(self.processed_splits_dir)
        self.process_spilts()

        path = osp.join(self.processed_splits_dir, 'pre_transform.pt')
        torch.save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_splits_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        print('Done!', file=sys.stderr)


class SNGNNWebKB(WebKB):
    def __init__(self, root: str, name: str):
        super().__init__(root, name)

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]

        train_mask = torch.stack(train_masks, dim=0)
        val_mask = torch.stack(val_masks, dim=0)
        test_mask = torch.stack(test_masks, dim=0)

        # train_mask = torch.stack(train_masks, dim=1)
        # val_mask = torch.stack(val_masks, dim=1)
        # test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class SNGNNWikipediaNetwork(WikipediaNetwork):
    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True):
        super().__init__(root, name, geom_gcn_preprocess)

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]

            train_mask = torch.stack(train_masks, dim=0).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=0).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=0).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            y = torch.from_numpy(data['target']).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class SNGNNActor(Actor):
    def __init__(self, root: str):
        super().__init__(root)

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]

        train_mask = torch.stack(train_masks, dim=0)
        val_mask = torch.stack(val_masks, dim=0)
        test_mask = torch.stack(test_masks, dim=0)

        # train_mask = torch.stack(train_masks, dim=1)
        # val_mask = torch.stack(val_masks, dim=1)
        # test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])



