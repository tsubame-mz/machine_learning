import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self, observation_space, action_space, hid_num):
        super(PNet, self).__init__()

        self.in_features = observation_space.n
        out_features = action_space.n

        # Policy : 各行動の確率を出力
        self.layer = nn.Sequential(
            nn.Linear(self.in_features, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, out_features),
            nn.Softmax(dim=-1),
        )

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.in_features).float()  # Onehot化
        return self.layer(x)


class VNet(nn.Module):
    def __init__(self, observation_space, hid_num):
        super(VNet, self).__init__()

        self.in_features = observation_space.n

        # Value : 状態の価値を出力
        self.layer = nn.Sequential(
            nn.Linear(self.in_features, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, 1),
        )

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.in_features).float()  # Onehot化
        return self.layer(x)
