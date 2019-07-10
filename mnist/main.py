import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import time


class AverageMeter:
    # 平均値測定器
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        assert self.count > 0
        return self.sum / self.count


def calc_accuracy(output_data, target_data):
    batch_size = target_data.size(0)
    _, pred_idx = output_data.topk(1, 1, True, True)  # 出力が最大のインデックスを取り出す
    pred_idx = pred_idx.t().squeeze().cpu()
    correct = pred_idx.eq(target_data)  # 正解: 1, 不正解: 0
    return (correct.sum().float() / batch_size) * 100.0


class Conv_BN_Act(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv_BN_Act, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.layers = nn.Sequential(
            # (1, 32, 32)
            Conv_BN_Act(1, 16, 3, stride=1, padding=1, bias=False),
            # (16, 32, 32)
            Conv_BN_Act(16, 16, 3, stride=1, padding=1, bias=False),
            # (16, 32, 32)
            nn.MaxPool2d(3, stride=1, padding=1),
            # (16, 32, 32)
            nn.AdaptiveAvgPool2d((1, 1)),
            # (16, 1, 1)
        )
        self.fc = nn.Linear(16, 10)
        # (10)

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 16)
        x = self.fc(x)
        return F.softmax(x, dim=-1)


def train(args, data_loader, model, loss_func, optimizer, epoch):
    # 学習モード
    model.train()

    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    batch_time_avg = AverageMeter()

    for i, (input_data, target_data) in enumerate(data_loader):
        start_time = time.perf_counter()

        output_data = model(input_data)
        loss = loss_func(output_data, target_data)
        loss_avg.update(loss.item(), target_data.size(0))

        acc = calc_accuracy(output_data.data, target_data)
        acc_avg.update(acc.item(), target_data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.perf_counter() - start_time
        batch_time_avg.update(batch_time)

        if ((i + 1) % args.log_interval) == 0:
            print(
                "Train: Epoch[{:03}/{:03}], Step[{:04}/{:04}], Loss[{:.4f}(Avg:{:.4f})], Acc[{:.3f}%(Avg:{:.3f}%)], Time[{:.3f}s(Avg:{:.3f}s)]".format(
                    epoch + 1,
                    args.max_epoch,
                    i + 1,
                    len(data_loader),
                    loss.item(),
                    loss_avg.avg,
                    acc.item(),
                    acc_avg.avg,
                    batch_time * args.log_interval,
                    batch_time_avg.avg * args.log_interval,
                )
            )
    print("Acc(Train)[{:.3f}%], Time[{:.3f}s]".format(acc_avg.avg, batch_time_avg.sum))


def validate(args, data_loader, model, loss_func, epoch):
    # 評価モード
    model.eval()

    loss_avg = AverageMeter()
    acc_avg = AverageMeter()
    batch_time_avg = AverageMeter()

    for i, (input_data, target_data) in enumerate(data_loader):
        start_time = time.perf_counter()

        output_data = model(input_data)
        loss = loss_func(output_data, target_data)
        loss_avg.update(loss.item(), target_data.size(0))

        acc = calc_accuracy(output_data.data, target_data)
        acc_avg.update(acc.item(), target_data.size(0))

        batch_time = time.perf_counter() - start_time
        batch_time_avg.update(batch_time)

        if ((i + 1) % args.log_interval) == 0:
            print(
                "Test: Epoch[{:03}/{:03}], Step[{:04}/{:04}], Loss[{:.4f}(Avg:{:.4f})], Acc[{:.3f}%(Avg:{:.3f}%)], Time[{:.3f}s(Avg:{:.3f}s)]".format(
                    epoch + 1,
                    args.max_epoch,
                    i + 1,
                    len(data_loader),
                    loss.item(),
                    loss_avg.avg,
                    acc.item(),
                    acc_avg.avg,
                    batch_time * args.log_interval,
                    batch_time_avg.avg * args.log_interval,
                )
            )
    print("Acc(Test)[{:.3f}%], Time[{:.3f}s]".format(acc_avg.avg, batch_time_avg.sum))
    return acc_avg.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="../datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    # 前処理(学習用)
    transform_train = transforms.Compose([transforms.ToTensor()])
    # 前処理(検証用)
    transform_validate = transforms.Compose([transforms.ToTensor()])

    # データセットの取得
    train_data_set = datasets.MNIST(args.datasets_dir, train=True, transform=transform_train, download=True)
    validate_data_set = datasets.MNIST(args.datasets_dir, train=False, transform=transform_validate, download=True)

    # データローダーの割り当て
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    validate_data_loader = torch.utils.data.DataLoader(
        validate_data_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    # モデル定義
    model = SimpleConvNet()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.9, nesterov=True)

    best_acc = 0
    for epoch in range(args.max_epoch):
        train(args, train_data_loader, model, loss_func, optimizer, epoch)
        epoch_acc = validate(args, validate_data_loader, model, loss_func, epoch)

        # is_best = epoch_acc > best_acc
        best_acc = max(epoch_acc, best_acc)

    print("Best accuracy[{:.3f}%]".format(best_acc))


if __name__ == "__main__":
    main()
