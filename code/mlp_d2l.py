import torch
import torchvision
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


batch_size, num_epochs, lr = 256, 3, 100.0
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_hiddens = 256
num_outputs = 10

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens))).float()
b1 = torch.zeros(num_hiddens).float()
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs))).float()
b2 = torch.zeros(num_outputs).float()

params = [W1, b1, W2, b2]
for param in params:
	param.requires_grad_(requires_grad=True)

def net(X):
	X = X.view(-1, num_inputs)
	H = torch.relu(torch.mm(X, W1) + b1)
	return torch.mm(H, W2) + b2

def relu(X):
	return torch.max(X, torch.tensor(0.0))

loss = nn.CrossEntropyLoss()

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, optimizer=None)