import torch
import torch.utils.data as Data 
import torch.nn as nn
import torch.optim as optim 

import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import d2lzh_pytorch as d2l
d2l.set_figsize()

#make data
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
dataset = Data.TensorDataset(features, labels)
batch_size = 10
data_itr = Data.DataLoader(dataset)

# model
num_epochs = 4
net = nn.Sequential(
	nn.Linear(num_inputs, 1)
	)
nn.init.normal_(net[0].weight, mean = 0, std=0.01)
nn.init.constant_(net[0].bias, val = 0)
Loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr = 1e-3)

for epoch in range(num_epochs):
	for X, y in data_itr:
		loss = Loss(net(X), y.view(-1, 1))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print('epoch %d, loss %f' % (epoch + 1, loss.item()))

print(net[0].weight, net[0].bias)