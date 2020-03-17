import torch
import torch.nn as nn

def corr2d(X, K):
    h, w= K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
            
    return Y

# =============================================================================
# test corr2d
# X = torch.arange(9).view(3, 3)         
# K = torch.arange(4).view(2, 2)
# print(corr2d(X, K))
# =============================================================================

# detect edge in images

X = torch.ones(6, 8)
X[:, 2:6] = 0
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
        
    def forward(self, X):
        return corr2d(X, self.weight) + self.bias


conv2d = Conv2D(kernel_size=(1, 2))
lr = 1e-2
epochs = 30

for epoch in range(epochs):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y)**2).sum()
    l.backward()
    
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    
    conv2d.weight.grad.data.zero_()
    conv2d.bias.grad.data.zero_()
    
    if (epoch+1)%5 == 0:
        print('Step %d, loss %.3f' % (epoch + 1, l.item()))
        
print(conv2d.weight, conv2d.bias)















