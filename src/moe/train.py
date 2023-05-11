import time
import warnings

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
pylogger = logging.getLogger(__name__)

def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')

    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')

    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')

    parser.add_argument('-e',
                        '--epochs',
                        default=2,
                        type=int,
                        help='number of total epochs (default: 30)')

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help="output logging information at a given interval")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    print('VAMOOOOOOOOOS')

    args = add_argument()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='/home/aghinassi/Desktop/nn-lab/src/moe/data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    testset = torchvision.datasets.CIFAR10(root='/home/aghinassi/Desktop/nn-lab/src/moe/data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # Encapsulate the model on the GPU assigned to the current process
    print('build model ...')
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset)

    fp16 = model_engine.fp16_enabled()
    device = model_engine.local_rank

    print('build metrics and losses ...')
    loss_fn = nn.CrossEntropyLoss()

    # Loop over the dataset multiple times
    print("Global Rank {} - Local Rank {} - Start Training ...".format(rank, local_rank))

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        if fp16:
            inputs = inputs.half()

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model_engine(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        model_engine.backward(loss)
        model_engine.step()

        # print the loss and accuracy metrics very log_interval mini-batches
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if i % args.log_interval == (args.log_interval - 1):
            print('[epoch %d, iterations %5d] loss: %.3f accuracy: %2f %%' % (
                epoch, i + 1, running_loss / args.log_interval, 100. * correct / total))
            running_loss = 0.0

    print('Training Done')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if fp16:
                images = images.half()
            outputs = model_engine(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            c = (predicted == labels.to(device)).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %2f %%' %
          (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2f %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))

    print('Evaluation Done')

if __name__ == "__main__":
    main()
