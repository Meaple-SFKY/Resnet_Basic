import torch
import torch.nn as nn
from torch.nn import init
import math

class Linear(object):
    def __init__(self, shape, in_features, out_features, if_bias):
        self.input_shape = shape
        self.batch_size = shape[0]
        self.in_features = in_features
        self.out_features = out_features
        self.if_bias = if_bias
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        self.bias = None
        if if_bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)