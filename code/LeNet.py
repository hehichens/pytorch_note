# -*- coding: utf-8 -*-
import torch 
from torch import nn, optim
import d2lzh_pytorch as d2l
import time 

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
            )
        
    def forward(self, X):
        feature = self.conv(X)
        output = self.fc(feature.view(X.shape[0], -1)) # add a dimension
        return output
    
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# =============================================================================
# test LeNet
# for X, y in train_iter:
#     X = X
#     y = y
#     break 
# net = LeNet()
# 
# print(net(X).shape)
# =============================================================================

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
    return acc_sum / n
    

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, num_epochs):
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0, 0, 0, 0, time.time()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc:%.3f, test acc:%.3f, time:%.1f sec"
              %(epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time() - start))

       
# =============================================================================
# test train_ch5
# net = LeNet()
# lr, num_epochs = 1e-3, 5
# optimizer = optim.Adam(net.parameters(), lr=lr)
# train_ch5(
#     net=net,
#     train_iter=train_iter,
#     test_iter=test_iter,
#     batch_size=batch_size,
#     optimizer=optimizer,
#     num_epochs=num_epochs
#     )
# =============================================================================















