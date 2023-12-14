from numpy import reshape
import torch
from torch.nn import init
import math

class Linear(object):
    def __init__(self, shape, in_features, out_features, if_bias=True):
        self.input_shape = shape
        self.batch_size = shape[0]
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = if_bias
        
        self.weight = torch.empty((in_features, out_features))
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        self.bias = None
        if if_bias:
            self.bias = torch.empty(out_features)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
        self.out_shape = [self.batch_size, self.out_features]
        self.weight_gradient = torch.zeros(self.weight.shape)
        if self.if_bias:
            self.bias_gradient = torch.zeros(self.bias.shape)
    
    def nn_forward_linear_func(self, x, weight):
        return torch.mm(x, weight) + self.bias
    
    def forward(self, x):
        self.x = x.reshape([self.batch_size, -1])
        return self.nn_forward_linear_func(self.x, self.weight)
    
    def gradient(self, eta):
        for b_idx in range(eta.shape[0]):
            col_x = self.x[b_idx].reshape(-1, 1)
            col_eta = eta[b_idx].reshape(1, -1)
            self.weight_gradient += torch.mm(col_x, col_eta)
            self.bias_gradient += col_eta.reshape(self.bias_gradient.shape)
        
        next_eta = torch.mm(eta, self.weight.T).reshape(self.input_shape)
        return next_eta
    
    def backward(self, alpha=0.00001, weight_decay=0.0004):
        self.weight = (1 - weight_decay) * self.weight - alpha * self.weight_gradient
        self.weight_gradient = torch.zeros(self.weight.shape)
        if self.if_bias:
            self.bias = (1 - weight_decay) * self.bias - alpha * self.bias_gradient
            self.bias_gradient = torch.zeros(self.bias.shape)
            

if __name__ == '__main__':
    # input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                         [2.0, 3.0, 4.0, 5.0],
    #                         [3.0, 4.0, 5.0, 6.0],
    #                         [4.0, 5.0, 6.0, 7.0]]]])
    input = torch.rand((2, 2, 4, 4))

    linear = Linear(input.shape, 32, 2, True)
    
    out = linear.forward(input)
    eta = linear.gradient(out)
    print(eta)
    linear.backward()