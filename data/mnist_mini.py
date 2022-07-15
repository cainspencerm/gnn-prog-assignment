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
        super().__init__('MNIST', raw_dir, force_reload, verbose)

        self._edges = []
        with open(os.path.join(raw_dir, 'edge_list.txt'), 'r') as f:
            for line in f.readlines():
                u, v = line.split(' ')
                u, v = int(u), int(v)
                self._edges.append((u, v))

        with open(os.path.join(raw_dir, 'zip_train.txt'), 'r') as f:
            lines = f.readlines()
        self._train_len = len(lines)

        with open(os.path.join(raw_dir, 'zip_test.txt'), 'r') as f:
            lines += f.readlines()
        self._test_len = len(lines) - self._train_len

        self._data, self._labels = [], []
        for line in lines:
            label, data = line.split(' ', maxsplit=1)

            data = np.array(list(map(float, data.split())))
            data = data.reshape(16, 16)

            self._data.append(data)
            self._labels.append(int(float(label)))

    def process(self):
        # Generate masks.
        train_mask = torch.tensor(np.array([True for i in range(len(self._labels)) if i < self._train_len]), dtype=torch.bool)
        test_mask = torch.tensor(np.array([True for i in range(len(self._labels)) if i >= self._train_len]), dtype=torch.bool)

        # build graph
        g = dgl.graph(self.edges)
        # splitting masks
        g.ndata['train_mask'] = train_mask
        g.ndata['test_mask'] = test_mask
        # node labels
        g.ndata['label'] = torch.tensor(self._labels)
        # node features
        g.ndata['feat'] = torch.tensor(self._data)
        self._num_labels = 10
        # reorder graph to obtain better locality.
        self._g = dgl.reorder_graph(g)

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1  # This dataset has only one graph

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        dgl.save_graphs(graph_path, self._g, {'labels': self._labels})

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self.graphs, label_dict = dgl.load_graphs(graph_path)
