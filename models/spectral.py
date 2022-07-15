from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn import DenseChebConv


class Classifier(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.gc1 = DenseChebConv(256, 64, 2)
        self.gc2 = DenseChebConv(64, 10, 2)
        self.dropout = dropout


        # Determined in parameter search.
        self.defaults = {'epochs': 20, 'batch_size': 32, 'learning_rate': 5e-3}

    def forward(self, x, adj, training=True):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


    def get_defaults(self):
        return 'epochs_' + str(self.defaults['epochs']) + \
            '_batch_size_' + str(self.defaults['batch_size']) + \
            '_learning_rate_' + '{:.0e}'.format(self.defaults['learning_rate'])


