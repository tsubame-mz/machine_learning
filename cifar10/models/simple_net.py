import torch.nn as nn
import torch.nn.functional as F

from models.octave_conv import OctaveConv


class Conv_BN_Act(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False):
        super(Conv_BN_Act, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class OctConv_BN_Act(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride=1, padding=1, bias=False):
        super(OctConv_BN_Act, self).__init__()

        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding)

        l_out_channels = int(alpha_out * out_channels)
        h_out_channels = out_channels - l_out_channels
        self.bn_h = nn.BatchNorm2d(h_out_channels)
        self.bn_l = nn.BatchNorm2d(l_out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l


class SimpleConvNet(nn.Module):
    def __init__(self, innter_channnels):
        super(SimpleConvNet, self).__init__()
        self.layers = nn.Sequential(
            # (N, C, H, W)
            Conv_BN_Act(3, innter_channnels, 3),
            Conv_BN_Act(innter_channnels, innter_channnels, 3),
            Conv_BN_Act(innter_channnels, innter_channnels, 3),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            Conv_BN_Act(innter_channnels, innter_channnels * 2, 3),
            Conv_BN_Act(innter_channnels * 2, innter_channnels * 2, 3),
            Conv_BN_Act(innter_channnels * 2, innter_channnels * 2, 3),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            Conv_BN_Act(innter_channnels * 2, innter_channnels * 4, 3),
            Conv_BN_Act(innter_channnels * 4, innter_channnels * 4, 3),
            Conv_BN_Act(innter_channnels * 4, innter_channnels * 4, 3),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(innter_channnels * 4, 10)

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=-1)


class SimpleOctConvNet(nn.Module):
    def __init__(self, innter_channnels):
        super(SimpleOctConvNet, self).__init__()
        alpha = 0.25
        self.layer1 = nn.Sequential(
            OctConv_BN_Act(3, innter_channnels, 3, 0, alpha),
            OctConv_BN_Act(innter_channnels, innter_channnels, 3, alpha, alpha),
            OctConv_BN_Act(innter_channnels, innter_channnels, 3, alpha, alpha),
        )
        self.layer2 = nn.Sequential(
            OctConv_BN_Act(innter_channnels, innter_channnels * 2, 3, alpha, alpha),
            OctConv_BN_Act(innter_channnels * 2, innter_channnels * 2, 3, alpha, alpha),
            OctConv_BN_Act(innter_channnels * 2, innter_channnels * 2, 3, alpha, alpha),
        )
        self.layer3 = nn.Sequential(
            OctConv_BN_Act(innter_channnels * 2, innter_channnels * 4, 3, alpha, alpha),
            OctConv_BN_Act(innter_channnels * 4, innter_channnels * 4, 3, alpha, alpha),
            OctConv_BN_Act(innter_channnels * 4, innter_channnels * 4, 3, alpha, 0),
        )
        self.layer4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.AdaptiveAvgPool2d((1, 1)))
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.fc = nn.Linear(innter_channnels * 4, 10)

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_h, x_l = self.layer1(x)
        x_h, x_l = self.downsample(x_h), self.downsample(x_l)
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.downsample(x_h), self.downsample(x_l)
        x_h, _ = self.layer3((x_h, x_l))
        x = self.layer4(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
