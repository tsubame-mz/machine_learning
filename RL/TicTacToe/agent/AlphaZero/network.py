import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        input_num = 10  # TODO
        hid_num = 32  # TODO
        out_Num = 9  # TODO

        self.common_layers = nn.Sequential(
            nn.Linear(input_num, hid_num), nn.ReLU(inplace=True), nn.Linear(hid_num, hid_num), nn.ReLU(inplace=True),
        )
        self.policy_layers = nn.Sequential(nn.Linear(hid_num, out_Num),)
        self.value_layers = nn.Sequential(nn.Linear(hid_num, 1), nn.Tanh(),)

        self.common_layers.apply(weight_init)
        self.policy_layers.apply(weight_init)
        self.value_layers.apply(weight_init)

    def inference(self, x: torch.Tensor, mask: torch.Tensor = None):
        # print(x, mask)
        h = self.common_layers(x)
        policy = self.policy_layers(h)
        value = self.value_layers(h)
        if mask is not None:
            policy += mask.masked_fill(mask == 1, -np.inf)
        return F.softmax(policy, dim=0), value

