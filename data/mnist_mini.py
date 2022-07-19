import torch
import numpy as np
from torch.utils.data import Dataset
import dgl
from dgl.data import DGLDataset
import os


class MNIST(Dataset):
    def __init__(self, data_dir=None, split='full'):
        super().__init__()

        assert split == 'train' or split == 'test' or split == 'full', 'split must be either train, test, or full'

        if split == 'full':
            with open(os.path.join(data_dir, 'zip_train.txt'), 'r') as f:
                lines = f.readlines()
            with open(os.path.join(data_dir, 'zip_test.txt'), 'r') as f:
                lines += f.readlines()
        else:
            with open(os.path.join(data_dir, f'zip_{split}.txt')) as f:
                lines = f.readlines()
        
        self.data, self.labels = [], []
        for line in lines:
            label, data = line.split(' ', maxsplit=1)

            data = np.array(list(map(float, data.split())))
            data = data.reshape(16, 16)

            self.data.append(data)
            self.labels.append(int(float(label)))

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float).unsqueeze(dim=0), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def get_balance(self):
        balance = np.zeros(len(set(self.labels)))
        for label in self.labels:
            balance[label] += 1
        return balance / np.sum(balance)


class MNIST_Graph(DGLDataset):
    def __init__(self, raw_dir='data', save_dir='data', force_reload=False, verbose=True):
        super().__init__('MNIST', raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    def process(self):
        row, col = [], []
        with open(os.path.join(self.raw_dir, 'edge_list.txt'), 'r') as f:
            for line in f.readlines():
                u, v = line.split(' ')
                u, v = int(u), int(v)
                row.append(u)
                col.append(v)
    
        self._edges = [(r, c) for r, c in zip(row, col)]
        self._edge_index = torch.tensor(np.array([row, col], dtype=np.int64), dtype=torch.long)

        with open(os.path.join(self.raw_dir, 'zip_train.txt'), 'r') as f:
            lines = f.readlines()
        self._train_len = len(lines)

        with open(os.path.join(self.raw_dir, 'zip_test.txt'), 'r') as f:
            lines += f.readlines()
        self._test_len = len(lines) - self._train_len

        self._data, self._labels = [], []
        for line in lines:
            label, data = line.split(' ', maxsplit=1)

            data = np.array(list(map(float, data.split())))

            self._data.append(data)
            self._labels.append(int(float(label)))

        self._data = np.array(self._data)
        self._labels = np.array(self._labels)

        # Generate masks.
        train_mask = torch.tensor(np.array([i < self._train_len for i in range(len(self._labels))]), dtype=torch.bool)
        test_mask = torch.tensor(np.array([i >= self._train_len for i in range(len(self._labels))]), dtype=torch.bool)

        # build graph
        g = dgl.graph(self._edges)
        # splitting masks
        g.ndata['train_mask'] = train_mask
        g.ndata['test_mask'] = test_mask
        # node labels
        g.ndata['label'] = torch.tensor(self._labels)
        # node features
        g.ndata['feat'] = torch.tensor(self._data, dtype=torch.float)
        self._num_labels = 10
        # reorder graph to obtain better locality.
        self._g = dgl.reorder_graph(g)

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1  # This dataset has only one graph

    def get_num_nodes(self, split='train'):
        if split == 'train':
            return self._train_len
        else:
            return self._test_len
    
    def edge_index(self):
        return self._edge_index
