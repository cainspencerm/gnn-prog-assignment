import torch
import numpy as np
from torch.utils.data import Dataset
import os


class MNIST(Dataset):
    def __init__(self, data_dir=None, split='train'):
        super().__init__()

        assert split == 'train' or split == 'test', 'split must be either train or test'
        
        self.data, self.labels = [], []
        with open(os.path.join(data_dir, f'zip_{split}.txt')) as f:
            for line in f.readlines():
                label, numbers = line.split(' ', maxsplit=1)
                self.data.append(np.array(list(map(float, numbers.split()))))
                self.labels.append(int(float(label)))

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)
