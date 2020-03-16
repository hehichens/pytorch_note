import torch
import torchvision
import torch.nn as nn
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import numpy as np

num_inputs, num_hiddens1, num_hiddens2, num_outputs = 784, 256, 256, 10

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Linear(num_hiddens2, num_outputs)
    )

optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)

num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

