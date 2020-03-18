# -*- coding: utf-8 -*-
import torch 
from torch import nn, optim
import d2lzh_pytorch as d2l


def batch_norm(is_training, X, gamma, beta, 
               moving_mean, moving_var, eps=1e-9, momentum=0.9):
    
    if is_training is not True:
        X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
        
    else:
        assert len(X.shape) in (2, 4)
        
        if len(X.shape)  == 2:
            mean = X.mean(dim=0)
            var = X.var(dim=0)
            
        else:
            mean = X.mean(dim=0, keepdim=True).mean(
                dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(
                dim=2, keepdim=True).mean(dim=3, keepdim=True)
        
        X_hat = (X - mean)/torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        shape = (1, num_features) if num_dims == 2 else (1, num_features, 1, 1)
        
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
        
    def forward(self, X):
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, 
        X, self.gamma, self.beta, self.moving_mean, self.moving_var)
        return Y

        
net = nn.Sequential(
           nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
           BatchNorm(6, num_dims=4),
           nn.Sigmoid(),
           nn.MaxPool2d(2, 2), # kernel_size, stride
           nn.Conv2d(6, 16, 5),
           BatchNorm(16, num_dims=4),
           nn.Sigmoid(),
           nn.MaxPool2d(2, 2),
           d2l.FlattenLayer(),
           nn.Linear(16*4*4, 120),
           BatchNorm(120, num_dims=2),
           nn.Sigmoid(),
           nn.Linear(120, 84),
           BatchNorm(84, num_dims=2),
           nn.Sigmoid(),
           nn.Linear(84, 10)
       )

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

lr, num_epochs = 0.001, 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)     
        
        
        
        
        
        