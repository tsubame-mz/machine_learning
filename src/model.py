import torch.nn as nn


class PolNet(nn.Module):
    def __init__(self, in_features, out_features, h1, h2):
        super(PolNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.layer(x)


class ValNet(nn.Module):
    def __init__(self, in_features, h1, h2):
        super(ValNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.layer(x)
