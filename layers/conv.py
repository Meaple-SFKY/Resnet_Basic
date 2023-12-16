import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from torch.nn import functional as F
from typing import Optional

class Conv2D(object):
    def __init__(self,
                 in_channels:  int = 1,
                 out_channels: int = 1,
                 kernel_size:  int = 3,
                 input_shape:  int = 4,
                 stride:       int = 1,
                 padding:      int = 0,
                 batchsize:    int = 1,
                 bias:        bool = False,
                 const_ker:   bool = False,
                 device:       str = 'cpu'
                 ) -> None:
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.input_shape  = input_shape
        self.stride       = stride
        self.padding      = padding
        self.batchsize    = batchsize
        self.if_bias      = bias
        self.device       = device

        if const_ker:
            self.weight = torch.tensor([[[[1.0, 2.0, 3.0],
                                          [2.0, 3.0, 4.0],
                                          [3.0, 4.0, 5.0]]]])
        else:
            self.weight = torch.empty((out_channels, in_channels, kernel_size, kernel_size), requires_grad=False).to(self.device)
            init.kaiming_normal_(self.weight, a=math.sqrt(5))
        
        if self.if_bias:
            self.bias   = torch.empty(out_channels, requires_grad=False).to(self.device)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        
        self.eta = torch.zeros((self.batchsize, self.out_channels, int(2 * self.padding + self.input_shape - self.kernel_size + 1), int(2 * self.padding + self.input_shape - self.kernel_size + 1)), requires_grad=False)
        self.weight_gradient = torch.zeros(self.weight.shape, requires_grad=False).to(self.device)
        self.sub_weight_gradient = torch.zeros(self.weight.shape, requires_grad=False).to(self.device)
        if self.if_bias:
            self.bias_gradient = torch.zeros(self.bias.shape, requires_grad=False).to(self.device)

    def nn_forward_conv2d_func(self, input: Tensor, weight: Tensor):
        return F.conv2d(input=input, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding)
    
    def nn_backward_conv2d_func(self, input: Tensor, weight: Tensor):
        return F.conv2d(input=input, weight=weight, bias=None, stride=self.stride, padding=self.padding)
    
    def forward(self, input: Tensor) -> Tensor:
        self.input_tensor = input
        out = self.nn_forward_conv2d_func(input, self.weight)
        return out
    
    def gradient(self, eta: Tensor) -> Tensor:
        self.eta = eta
        for b_idx in range(self.batchsize):
            torch.zero_(self.sub_weight_gradient)
            for o_idx in range(self.out_channels):
                sub_eta = eta[b_idx][o_idx].reshape(1, 1, eta.shape[2], eta.shape[3])
                for i_idx in range(self.in_channels):
                    sub_input = self.input_tensor[b_idx][i_idx].reshape(1, 1, self.input_shape, self.input_shape)
                    tmp_out = F.conv2d(sub_input, sub_eta).reshape(self.sub_weight_gradient.shape[2], self.sub_weight_gradient.shape[3])
                    self.sub_weight_gradient[o_idx][i_idx] = tmp_out
            self.weight_gradient += self.sub_weight_gradient
            if self.if_bias:
                self.bias_gradient += eta[b_idx].sum(dim=(1, 2))
        
        pad_shape = (self.kernel_size - 1, self.kernel_size - 1, self.kernel_size - 1, self.kernel_size - 1, 0, 0, 0, 0)
        pad_eta = F.pad(eta, pad_shape, 'constant', 0)
        fliped_weight = torch.flip(self.weight, dims=(2, 3)).swapaxes(0, 1)
        next_eta = self.nn_backward_conv2d_func(pad_eta, fliped_weight)

        return next_eta
    
    def backward(self, alpha = 0.000001, decay = 0.00001) -> None:
        self.weight = self.weight * (1 - decay) - alpha * self.weight_gradient
        torch.zero_(self.weight_gradient)
        if self.if_bias:
            self.bias = self.bias * (1 - decay) - alpha * self.bias_gradient
            torch.zero_(self.bias_gradient)

if __name__ == "__main__":
    # input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                         [2.0, 3.0, 4.0, 5.0],
    #                         [3.0, 4.0, 5.0, 6.0],
    #                         [4.0, 5.0, 6.0, 7.0]]]])
    input = torch.rand((2, 2, 5, 5))

    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, input_shape=5, stride=1, padding=0, batchsize=2, bias=True, const_ker=False)

    output = conv.forward(input)
    label = output + 1
    loss = nn.CrossEntropyLoss()
    l = loss(output, label)
    print(l)
    print(conv.weight)
    eta = conv.gradient(output)
    conv.backward()
    print(conv.weight)