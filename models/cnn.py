from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2880, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x, training=True):
        x = self.conv1(x)
        x = self.conv2(x)

        if training:
            x = self.conv2_drop(x)

        x = x.view(-1, 2880)
        x = self.fc1(x)
        x = self.fc2(x)
        return x