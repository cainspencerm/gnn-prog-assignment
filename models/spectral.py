from torch import nn
from dgl import nn as dglnn

# Determined in parameter search.
defaults = {'epochs': 17, 'learning_rate': 5e-3}

def get_defaults():
    return 'epochs_' + str(defaults['epochs']) + \
        '_learning_rate_' + '{:.0e}'.format(defaults['learning_rate'])


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.gc1 = dglnn.DenseChebConv(256, 256, 2)
        self.gc2 = dglnn.DenseChebConv(256, 256, 2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, adj, x):
        x = self.gc1(adj, x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.gc2(adj, x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


