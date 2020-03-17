import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AlphaZeroConfig


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


class SwishImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishImpl.apply(x)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ResNetBlock(nn.Module):
    def __init__(self, num_channels, activation=Swish):
        super(ResNetBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            activation(),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
        )
        self.relu = activation()

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.relu(x + self.layers(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, num_channels, activation=Swish):
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_channels, num_channels),
            nn.BatchNorm2d(num_channels),
            activation(),
            ResNetBlock(num_channels, activation),
            ResNetBlock(num_channels, activation),
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, input_Num, hid_num, output_num, activation=Swish):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_Num, hid_num),
            activation(),
            nn.Linear(hid_num, hid_num),
            activation(),
            nn.Linear(hid_num, output_num),
        )
        self.layers.apply(weight_init)

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.layers(x)


class AlphaZeroNetwork(nn.Module):
    def __init__(self, config: AlphaZeroConfig, activation=Swish):
        super(AlphaZeroNetwork, self).__init__()

        self.num_channels = config.num_channels
        in_channels = config.obs_space[0]
        self.resnet = ResNet(in_channels, config.num_channels, activation)

        ch_h = config.obs_space[1]
        ch_w = config.obs_space[2]
        self.fc_input_num = config.num_channels * ch_h * ch_w
        self.policy_layers = MLP(self.fc_input_num, config.fc_hid_num, config.fc_output_num, activation)
        self.value_layers = MLP(self.fc_input_num, config.fc_hid_num, config.atoms, activation)

    def inference(self, x: torch.Tensor):
        h = self.resnet(x)
        h = h.view(-1, self.fc_input_num)
        return self.policy_layers(h), self.value_layers(h)


if __name__ == "__main__":
    config = AlphaZeroConfig()
    config.obs_space = (3, 3, 3)
    config.num_channels = 8
    config.fc_hid_num = 16
    config.fc_output_num = 9
    config.support_size = 10
    config.atoms = config.support_size * 2 + 1
    batch_size = 4

    network = AlphaZeroNetwork(config)
    print(network)
    x = torch.randn((batch_size, config.obs_space[0], config.obs_space[1], config.obs_space[2]))
    print(x)
    policy_logits, value_logits = network.inference(x)
    print(policy_logits)
    print(value_logits)
