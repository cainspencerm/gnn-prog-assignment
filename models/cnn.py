from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv_drop = nn.Dropout2d(0.3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
        # Determined in parameter search.
        self.defaults = {'epochs': 6, 'batch_size': 32, 'learning_rate': 1e-2}

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv_drop(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv_drop(x)

        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x

    def get_defaults(self):
        return 'epochs_' + str(self.defaults['epochs']) + \
            '_batch_size_' + str(self.defaults['batch_size']) + \
            '_learning_rate_' + '{:.0e}'.format(self.defaults['learning_rate'])