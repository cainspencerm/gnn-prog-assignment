from torch import nn
from dgl import nn as dglnn


# Determined in parameter search.
defaults = {'epochs': 30, 'learning_rate': 5e-3}

def get_defaults():
    return 'epochs_' + str(defaults['epochs']) + \
        '_learning_rate_' + '{:.0e}'.format(defaults['learning_rate'])

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = dglnn.GATConv(256, 256, num_heads=3, attn_drop=0.5, residual=True)
        self.gat2 = dglnn.GATConv(256, 256, num_heads=3, attn_drop=0.5, residual=True)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
       
       # Determined in parameter search.
        self.defaults = {'epochs': 20, 'batch_size': 32, 'learning_rate': 5e-3}

    def forward(self, graph, x):
        x = self.gat1(graph, x)
        x = self.elu(x)
        x = self.dropout(x)

        x = self.gat2(graph, x)
        x = self.elu(x)
        x = self.dropout(x)

        x = x.view(-1, 2304)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def get_defaults(self):
        return 'epochs_' + str(self.defaults['epochs']) + \
            '_batch_size_' + str(self.defaults['batch_size']) + \
            '_learning_rate_' + '{:.0e}'.format(self.defaults['learning_rate'])