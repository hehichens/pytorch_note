import torch
import torchvision
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28*28
num_outputs = 10

net = nn.Sequential(
	nn.Flatten(),
	nn.Linear(num_inputs, num_outputs)
	)
nn.init.normal_(net[1].weight, mean=0, std=0.01)
nn.init.constant_(net[1].bias, val=0)

optimizer = optim.SGD(net.parameters(), lr=1e-2)
cross_entropy = nn.CrossEntropyLoss()

def softmax(X):
	X_exp = X.exp()
	partition = X_exp.sum(dim=1, keepdim=True)
	return X_exp / partition


def accuracy(y_hat, y):
	return (y_hat.argmax(dim = 1) == y).float().sum().item()

def evaluate_accuracy(data_iter, net):
	acc_sum, n = 0, 0
	for X, y in data_iter:
		acc_sum += (net(X).argmax(dim = 1) == y).float().sum().item()
		n += y.shape[0]
	return acc_sum / n


def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
		 params=None, lr=None, optimizer=None):
	for epoch in range(num_epochs):
		train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
		for X, y in train_iter:
			y_hat = net(X)
			l = loss(y_hat, y).sum()
			optimizer.zero_grad()

			l.backward()
			optimizer.step()  # “softmax回归的简洁实现”一节将用到

			train_l_sum += l.item()
			train_acc_sum += accuracy(y_hat, y)
			n += y.shape[0]
		test_acc = evaluate_accuracy(test_iter, net)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

num_epochs = 5

train(
	net=net,
	train_iter = train_iter,
	test_iter = test_iter,
	loss = cross_entropy,	
	num_epochs = num_epochs,
	batch_size = batch_size,
	params = None,
	lr = None,
	optimizer = optimizer
	)
