import torch
import torchvision
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import numpy as np


n, true_w, true_b = 200, [1.2, -3.4, 5.6], 5
features = torch.randn(n, 1)
poly_features = torch.cat((features, features**2, features**3), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

num_epochs, loss = 100, torch.nn.MSELoss()

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
	plt.figure()
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.semilogy(x_vals, y_vals)
	if x2_vals and y2_vals:
	    plt.semilogy(x2_vals, y2_vals, linestyle=':')
	    plt.legend(legend)
    


def fit_and_plot(train_features, test_features, train_labels, test_labels):
	net = torch.nn.Linear(train_features.shape[-1], 1)
	batch_size = min(10, train_labels.shape[0])
	
	dataset = torch.utils.data.TensorDataset(train_features, train_labels)
	train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

	optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
	train_ls, test_ls = [], []
	for _ in range(num_epochs):
		for X, y in train_iter:
			l = loss(net(X), y.view(-1, 1))
			optimizer.zero_grad()
			l.backward()
			optimizer.step()

		train_labels = train_labels.view(-1, 1)
		test_labels = test_labels.view(-1, 1)

		train_ls.append(loss(net(train_features), train_labels).item())
		test_ls.append(loss(net(test_features), test_labels).item())

	semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])

	print('weight:', net.weight.data,
	      '\nbias:', net.bias.data)

# 正常拟合
n_train = 150
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], 
            labels[:n_train], labels[n_train:])


# 欠拟合
fit_and_plot(features[:n_train, :], features[n_train:, :], 
            labels[:n_train], labels[n_train:])


# 过拟合
n_train = 50
fit_and_plot(features[:n_train, :], features[n_train:, :], 
            labels[:n_train], labels[n_train:])

plt.show()