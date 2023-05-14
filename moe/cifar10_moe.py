import argparse

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU)"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        fc3 = Mlp_GEGLU(84, 128, 84)
        self.experts = deepspeed.moe.layer.MoE(hidden_size=84, expert=fc3,
                                               num_experts=4, ep_size=2, k=2)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, _, _ = self.experts(x)
        x = self.fc4(x)
        return x

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def run(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()

    trainset = torchvision.datasets.CIFAR10(root='/home/aghinassi/Desktop/nn-lab/moe/data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=trainset)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=2,
                                              sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(root='/home/aghinassi/Desktop/nn-lab/moe/data',
                                           train=False,
                                           download=True,
                                           transform=transform)

    val_sampler = DistributedSampler(dataset=testset)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2,
                                             sampler=val_sampler)

    net = Net()
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(model_engine.local_rank)

            # Forward pass
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            # print the loss and accuracy metrics very log_interval mini-batches
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if i % args.log_interval == (200 - 1):
                print('[epoch %d, iterations %5d] loss: %.3f accuracy: %2f %%' % (
                    epoch, i + 1, running_loss / 200, 100. * correct / total))
                running_loss = 0.0

        evaluate(model_engine, testloader)

    print('Training Done')
    return running_loss

@torch.no_grad()
def evaluate(model_engine, testloader):
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(model_engine.local_rank)).sum().item()
        c = (predicted == labels.to(model_engine.local_rank)).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    print(f'Accuracy of the network on the {total} test images: %2f %%' %
          (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2f %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))

    print('Evaluation Done')

def main():
    try:
        args = add_argument()
        run(args)
    except Exception as e:
        dist.destroy_process_group()
        raise e
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
