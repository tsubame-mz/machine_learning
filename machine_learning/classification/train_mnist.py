import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc1 = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    mnist_path = "../../data"
    batch_size = 4

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    train_datasets = torchvision.datasets.MNIST(mnist_path, train=True, download=True, transform=transform)
    test_datasets = torchvision.datasets.MNIST(mnist_path, train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=1)

    net = Net(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Start Training")
    net.train()
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            x, t = data
            y = net(x)
            loss = criterion(y, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print(f"[{epoch + 1:2d}, {i + 1:5d}] loss: {running_loss / 1000:.4f}")
                running_loss = 0.0
    print("Finish Training")

    print("Start Test")
    net.eval()
    correct = 0
    total = 0
    class_correct = list(0 for i in range(10))
    class_total = list(0 for i in range(10))
    with torch.no_grad():
        for data in test_dataloader:
            x, t = data
            y = net(x)
            _, predicted = torch.max(y, 1)
            total += t.size(0)
            correct += (predicted == t).sum().item()
            c = (predicted == t).squeeze()
            for t_data, c_data in zip(t, c):
                class_correct[t_data] += c_data.item()
                class_total[t_data] += 1
    print(f"Total: {total}, Correct: {correct}, Accuracy: {100 * correct / total:6.2f}%")
    for i in range(10):
        print(
            f"Class: {i}, Total, {class_total[i]:4d}, Correct: {class_correct[i]:4d}, Accuracy: {100 * class_correct[i] / class_total[i]:6.2f}%"
        )
    print("Finish Test")
