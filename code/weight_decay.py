import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import numpy as np


n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
	plt.figure()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.semilogy(x_vals, y_vals)
	if x2_vals and y2_vals:
	    plt.semilogy(x2_vals, y2_vals, linestyle=':')
	    plt.legend(legend)
  
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]  

def l2_penalty(w):
	return (w**2).sum() / 2

batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    net = nn.Linear(num_inputs, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=1)
    optimizer_w = optim.SGD(params=[net.weight], lr=lr, weight_decay=lambd)
    optimizer_b = optim.SGD(params=[net.bias], lr=lr)
    train_ls, test_ls = [], []

    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X), y).mean()
            
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            optimizer_w.step()
            optimizer_b.step()

        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())


fit_and_plot(lambd=4)

plt.show()