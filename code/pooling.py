# -*- coding: utf-8 -*-
import torch 
from torch import nn

#pooling layer
def pool2d(X, pool_size, mod='max'):
    X = X.float()
    h, w = pool_size
    Y = torch.zeros(X.shape[0]+1-h, X.shape[1]+1-w)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mod =='max':
                Y[i, j] = X[i:i+h, j:j+w].max()
            if mod == 'avg':
                Y[i, j] = X[i:i+h, j:j+w].mean()
    return Y

# =============================================================================
# test pool2d
# X = torch.arange(9).view(3, 3)
# print(pool2d(X, pool_size=(2, 2), mod='max'))
# print(pool2d(X, pool_size=(2, 2), mod='avg'))                
# =============================================================================

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
pool2d_1 = nn.MaxPool2d(3, padding=1, stride=2)
# =============================================================================
# print(pool2d_1(X))
# =============================================================================


pool2d_2 = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
# =============================================================================
# print(pool2d_2(X))
# =============================================================================


# multi channels
X = torch.cat((X, X + 1), dim=1)
pool2d_3 = nn.MaxPool2d(3, padding=1, stride=2)
# =============================================================================
# print(pool2d_3(X))
# =============================================================================



