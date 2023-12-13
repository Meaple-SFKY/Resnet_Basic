import torch

class Relu(object):
    def __init__(self, shape):
        self.eta = torch.zeros(shape)
        self.x = torch.zeros(shape)
    
    def forward(self, x):
        self.x = x
        return torch.maximum(x, torch.tensor(0))
    
    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x < 0] = 0
        return self.eta
    
if __name__ == "__main__":
    # input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                         [2.0, 3.0, 4.0, 5.0],
    #                  cd ..       [3.0, 4.0, 5.0, 6.0],
    #                         [4.0, 5.0, 6.0, 7.0]]]])
    input = torch.rand((2, 2, 5, 5)) - 0.5

    relu = Relu(input.shape)
    print(input)
    out = relu.forward(input)
    print(out)
    grad = relu.gradient(out)
    print(grad)