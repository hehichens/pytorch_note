import torch
import torchvision
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


batch_size, num_epochs= 256, 3

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_hiddens = 256
num_outputs = 10

net = nn.Sequential(
	nn.Flatten(),
	nn.Linear(num_inputs, num_hiddens),
	nn.ReLU(),
	nn.Linear(num_hiddens, num_outputs)
	)

for param in net.parameters():
	nn.init.normal_(param, mean=0, std=0.01)

optimizer = optim.SGD(net.parameters(), lr=0.5)

loss = nn.CrossEntropyLoss()

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)