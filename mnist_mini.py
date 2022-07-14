import torch
import numpy as np
from torch.utils.data import Dataset
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
            label, numbers = line.split(' ', maxsplit=1)

            data = np.array(list(map(float, numbers.split())))
            data = data.reshape(16, 16)

            self.data.append(data)
            self.labels.append(int(float(label)))

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float).unsqueeze(dim=0), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)
