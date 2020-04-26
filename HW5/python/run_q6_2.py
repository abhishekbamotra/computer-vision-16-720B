# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 02:52:14 2020

@author: pbamo
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from run_q6_1 import test, train
import multiprocessing


class Scratch(nn.Module):
    def __init__(self):
        super(Scratch, self).__init__()
        self.conv_1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv_2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv_3 = nn.Conv2d(20, 40, kernel_size=3)
        self.fully_connect_1 = nn.Linear(27040, 256)
        self.fully_connect_2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_1(x), 2))
        x = F.relu(F.max_pool2d(self.conv_2(x), 2))
        x = F.relu(F.max_pool2d(self.conv_3(x), 2))
        x = x.view(-1, 27040)
        x = F.relu(self.fully_connect_1(x))
        x = F.dropout(x, training=self.training)
        x = self.fully_connect_2(x)
        result = F.log_softmax(x, dim=1)
        return result



if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} if use_cuda else {}
    max_iters = 50

    #Data loading
    transform_data = transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224),transforms.ToTensor()])

    train_loader = DataLoader(ImageFolder('../data/oxford-flowers17/train', transform = transform_data), batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(ImageFolder('../data/oxford-flowers17/val', transform = transform_data), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(ImageFolder('../data/oxford-flowers17/test', transform = transform_data), batch_size=args.batch_size, shuffle=True, **kwargs)



    ##### Scratch ######
    model = Scratch().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_loss, train_acc = list(), list()
    for i in range(max_iters):
        loss, acc = train(args, model, device, train_loader, optimizer, i)
        train_loss.append(loss)
        train_acc.append(acc)
        print('Iteration: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(i, loss, acc))
    test(args, model, device, test_loader)
    
    x = np.arange(max_iters)
    plt.figure('Accuracy')
    plt.plot(x, train_acc)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations - Scratch')
    plt.show()
    
    
    ##### Fine Tuning ######
    
    model = models.squeezenet1_1(pretrained=True)
    # print (model)
    num_classes = len(ImageFolder('../data/oxford-flowers17/train', transform = transform_data).classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model.num_classes = num_classes

    model.type(torch.FloatTensor)
    cross_loss = nn.CrossEntropyLoss().type(torch.FloatTensor)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    train_acc_fine = list()
    for i in range(max_iters):
        model.train()
        for x, y in train_loader:
          x_var = Variable(x.type(torch.FloatTensor))
          y_var = Variable(y.type(torch.FloatTensor).long())
          scores = model(x_var)
          loss = cross_loss(scores, y_var)
      
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # Check accuracy on the train and val sets.
        model.eval()
        match, n_values = 0, 0
        for x, y in train_loader:
          x_var = Variable(x.type(torch.FloatTensor), volatile=True)
          scores = model(x_var)
          
          no_need, preds = scores.data.cpu().max(1)
          match += (preds == y).sum()
          n_values += x.size(0)

        train_acc_f = float(match) / n_values
        train_acc_fine.append(train_acc_f)
        print('Iteration: {}, Train accuracy: {}'.format(i, train_acc_f))
        
    x = np.arange(max_iters)
    plt.figure('Accuracy')
    plt.plot(x, train_acc_fine)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations - Fine Tuning')
    plt.show()
    
    
    model.eval()
    match, n_values = 0, 0
    for x, y in valid_loader:
      x_var = Variable(x.type(torch.FloatTensor), volatile=True)
  
      scores = model(x_var)
      no_need, preds = scores.data.cpu().max(1)
      match += (preds == y).sum()
      n_values += x.size(0)

    print('Validation accuracy: ', float(match) / n_values)