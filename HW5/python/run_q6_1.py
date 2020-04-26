# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 02:18:33 2020

@author: abamo
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    cross_loss = nn.CrossEntropyLoss()
    total_loss, match = 0, 0
    for idx, (data, target) in enumerate(train_loader):
        
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = cross_loss(output, target)
        
        pred = output.max(1, keepdim=True)
        pred_out = pred[1].eq(target.view_as(pred[1]))
        match += pred_out.sum().item()
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()            
    
    # total_loss /= len(train_loader.dataset)
    acc = match / len(train_loader.dataset)
    return total_loss, acc

def test(args, model, device, test_loader):
    model.eval()
    test_loss, match = 0, 0
    cross_loss = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            test_loss += cross_loss(output, target).item()
            
            
            pred = output.max(1, keepdim=True)
            pred_out = pred[1].eq(target.view_as(pred[1]))
            
            match += pred_out.sum().item()

    test_acc = match / len(test_loader.dataset)
    print('Test Average loss: {:.6f},Test Accuracy: {:.06f}%'.format(test_loss, test_acc))

class Load_Nist(Dataset):
    def __init__(self, path, train=True, conv=False):
        data = scipy.io.loadmat(path)
        val = 'train' if train else 'test'
        data_x, data_y =  data[val+'_data'], data[val+'_labels']
        if conv:
            data_x = np.reshape(data_x, (data_x.shape[0],1,32,32))
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.argmax(torch.from_numpy(data_y), dim=1).long()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (self.data_x[idx], self.data_y[idx])
    
class FullyConnect(nn.Module):
    def __init__(self):
        super(FullyConnect, self).__init__()
        self.fully_connect1 = nn.Linear(1024, 64)
        self.fully_connect_2 = nn.Linear(64, 36)

    def forward(self, x):
        x = torch.sigmoid(self.fully_connect1(x))
        x = self.fully_connect_2(x)
        result = F.log_softmax(x, dim=1)
        return result
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fully_connect1 = nn.Linear(500, 50)
        self.fully_connect_2 = nn.Linear(50, 36)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_2(x), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fully_connect1(x))
        x = F.dropout(x, training=self.training)
        x = self.fully_connect_2(x)
        result = F.log_softmax(x, dim=1)
        return result
    
class ConvCifar(nn.Module):
    def __init__(self):
        super(ConvCifar, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fully_connected_1 = nn.Linear(16 * 5 * 5, 120)
        self.fully_connected_2 = nn.Linear(120, 84)
        self.fully_connected_3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fully_connected_1(x))
        x = F.relu(self.fully_connected_2(x))
        result = self.fully_connected_3(x)
        return result


def main_network(conv=False, cifar=False):
    
    ## Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=50, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    
    
    # Data Loading
    if cifar:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform), batch_size=4,
                                              shuffle=True, num_workers=2)
        
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform), batch_size=4,
                                                 shuffle=False, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(Load_Nist(path='../data/nist36_train.mat', train=True, conv=conv),batch_size=args.batch_size, shuffle=True, **kwargs) 
        test_loader = torch.utils.data.DataLoader(Load_Nist(path='../data/nist36_test.mat', train=False, conv=conv),batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    
    
    #Check if conv or fully connected
    if conv and not cifar:
        model = ConvNet().to(device)
        max_iter = 50
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        
    elif conv and cifar:
        model = ConvCifar().to(device)
        max_iter = 50
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        model = FullyConnect().to(device)
        max_iter = 250
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


    #Training the model
    train_loss, train_acc = list(), list()
    
    for i in range(max_iter):
        loss, acc = train(args, model, device, train_loader, optimizer, i)
        print('Iteration: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(i, loss, acc))
        
        train_loss.append(loss)
        train_acc.append(acc)
    
    test(args, model, device, test_loader)
    
    
    ## Plot accuracy and loss plot
    x = np.arange(max_iter)
    plt.figure('loss')
    plt.plot(x, train_loss)
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title('Loss vs Iterations')
    plt.show()
    
    plt.figure('Accuracy')
    plt.plot(x, train_acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.show()

if __name__ == '__main__':
    # Q 6.1.1
    main_network()
    
    # # Q 6.1.2
    main_network(conv=True)
    
    # Q 6.1.3
    main_network(conv=True, cifar=True)