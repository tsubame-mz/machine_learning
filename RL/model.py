import gym
import torch.nn as nn


class PVNet(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super(PVNet, self).__init__()

        hid_num = args.hid_num
        droprate = args.droprate

        # in
        if isinstance(observation_space, gym.spaces.Box):
            in_features = observation_space.shape[0]
        elif isinstance(observation_space, gym.spaces.Discrete):
            in_features = observation_space.n
        # out
        if isinstance(action_space, gym.spaces.Discrete):
            out_features = action_space.n
        print("in_features[{}], out_features[{}]".format(in_features, out_features))

        # 共通ネットワーク
        self.layer = nn.Sequential(
            nn.Embedding(in_features, hid_num),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
        )

        # Policy : 各行動の確率を出力
        self.policy = nn.Sequential(
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(hid_num, out_features),
            nn.Softmax(dim=-1),
        )

        # Value : 状態の価値を出力
        self.value = nn.Sequential(
            nn.Linear(hid_num, hid_num), nn.ReLU(inplace=True), nn.Dropout(p=droprate), nn.Linear(hid_num, 1)
        )

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.layer(x.long())
        return self.policy(h), self.value(h)
