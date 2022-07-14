from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()



        # Determined in parameter search.
        self.defaults = {'epochs': 20, 'batch_size': 32, 'learning_rate': 5e-3}

    def forward(self, x, training=True):
        return x

    def get_defaults(self):
        return 'epochs_' + str(self.defaults['epochs']) + \
            '_batch_size_' + str(self.defaults['batch_size']) + \
            '_learning_rate_' + '{:.0e}'.format(self.defaults['learning_rate'])


