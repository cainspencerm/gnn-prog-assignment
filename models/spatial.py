from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn import GATConv


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        """ two spatial layers:
        studied and referenced from the
        https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.GATConv.html
        """
        self.gat1 = GATConv(256, 64, 3)
        self.gat2 = GATConv(64, 10, 3)
        self.optimizer = optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)
       
       # Determined in parameter search.
        self.defaults = {'epochs': 20, 'batch_size': 32, 'learning_rate': 5e-3}

    def forward(self, x, edge_index, training=True):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def get_defaults(self):
        return 'epochs_' + str(self.defaults['epochs']) + \
            '_batch_size_' + str(self.defaults['batch_size']) + \
            '_learning_rate_' + '{:.0e}'.format(self.defaults['learning_rate'])