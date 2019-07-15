import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

from models.simple_net import SimpleConvNet
from models.simple_net import SimpleOctConvNet


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


def train(args, data_loader, model, loss_func, optimizer, epoch, writer, log_step):
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
            if writer:
                writer.add_scalar("train/loss", loss.item(), log_step)
                writer.add_scalar("train/acc", acc.item(), log_step)
                log_step += 1
    print("Acc(Train)[{:.3f}%], Time[{:.3f}s]".format(acc_avg.avg, batch_time_avg.sum))
    return log_step


def validate(args, data_loader, model, loss_func, epoch, writer, log_step):
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
                "Validate: Epoch[{:03}/{:03}], Step[{:04}/{:04}], Loss[{:.4f}(Avg:{:.4f})], Acc[{:.3f}%(Avg:{:.3f}%)], Time[{:.3f}s(Avg:{:.3f}s)]".format(
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
            if writer:
                writer.add_scalar("validate/loss", loss.item(), log_step)
                writer.add_scalar("validate/acc", acc.item(), log_step)
                log_step += 1
    print("Acc(Validate)[{:.3f}%], Time[{:.3f}s]".format(acc_avg.avg, batch_time_avg.sum))
    return acc_avg.avg, log_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="../datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--innter_channnels", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--use_logfile", type=bool, default=True)
    parser.add_argument("--use_octconv", type=bool, default=False)
    args = parser.parse_args()

    if args.use_logfile:
        if args.use_octconv:
            writer = SummaryWriter("../logs/cifar10-octconv")
        else:
            writer = SummaryWriter("../logs/cifar10")
    else:
        writer = None

    # 前処理(学習用)
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # 前処理(検証用)
    transform_validate = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # データセットの取得
    train_data_set = datasets.CIFAR10(args.datasets_dir, train=True, transform=transform_train, download=True)
    validate_data_set = datasets.CIFAR10(args.datasets_dir, train=False, transform=transform_validate, download=True)

    # データローダーの割り当て
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )
    validate_data_loader = torch.utils.data.DataLoader(
        validate_data_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    # モデル定義
    if args.use_octconv:
        model = SimpleOctConvNet(args.innter_channnels)
    else:
        model = SimpleConvNet(args.innter_channnels)
    print(model)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_acc = 0
    train_log_step = 0
    validate_log_step = 0
    for epoch in range(args.max_epoch):
        train_log_step = train(args, train_data_loader, model, loss_func, optimizer, epoch, writer, train_log_step)
        epoch_acc, validate_log_step = validate(
            args, validate_data_loader, model, loss_func, epoch, writer, validate_log_step
        )
        best_acc = max(epoch_acc, best_acc)

    print("Best accuracy[{:.3f}%]".format(best_acc))
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
