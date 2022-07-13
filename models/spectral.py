from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, training=True):
        return x