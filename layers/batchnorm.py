from cv2 import mean
import torch

class BatchNorm(object):
    def __init__(self, shape):
        self.out_shape = shape
        self.batch_size = shape[0]
        self.input_data = torch.zeros(shape)
        self.channels = shape[1]
        
        self.alpha = torch.ones(self.channels).reshape(1, -1, 1, 1)
        self.beta = torch.zeros(self.channels).reshape(1, -1, 1, 1)
        self.alpha_gradient = torch.zeros(self.channels).reshape(1, -1, 1, 1)
        self.beta_gradient = torch.zeros(self.channels).reshape(1, -1, 1, 1)
        
        self.moving_mean = torch.zeros(self.channels)
        self.moving_var = torch.zeros(self.channels)
        self.epsilon = 0.00001
        self.moving_decay = 0.997
    
    def forward(self, x, mode='train'):
        self.input_data = x
        
        if mode == 'train':
            self.mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            self.var = ((x - self.mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            self.normed_x = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
            if torch.sum(self.moving_mean) == 0 and torch.sum(self.moving_var) == 0:
                self.moving_mean = self.mean
                self.moving_var = self.var
            else:
                self.moving_mean = self.moving_decay * self.moving_mean + (1 - self.moving_decay) * self.mean
                self.moving_var = self.moving_decay * self.moving_var + (1 - self.moving_decay) * self.var
        else:
            self.normed_x = (x - self.moving_mean) / torch.sqrt(self.moving_var + self.epsilon)
        return self.normed_x * self.alpha + self.beta
    
    def gradient(self, eta):
        self.alpha_gradient = torch.sum(eta * self.normed_x, dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        self.bias_gradient = torch.sum(eta * self.normed_x, dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        
        normed_x_gradient = eta * self.alpha
        var_gradient = torch.sum(-1.0 / 2 * normed_x_gradient * (self.input_data - self.mean) / (self.var + self.epsilon) ** (3.0 / 2), dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        mean_gradient = torch.sum(-1.0 / torch.sqrt(self.var + self.epsilon) * normed_x_gradient, dim=(0, 2, 3)).reshape(1, -1, 1, 1)
        
        x_gradient = normed_x_gradient * torch.sqrt(self.var + self.epsilon) + 2 * (self.input_data - self.mean) * var_gradient / self.batch_size + mean_gradient / self.batch_size
        
        return x_gradient
    
    def backward(self, alpha=0.0001):
        self.alpha -= alpha * self.alpha_gradient
        self.beta -= alpha * self.beta_gradient
        self.alpha_gradient = torch.zeros(self.channels).reshape(1, -1, 1, 1)
        self.bias_gradient = torch.zeros(self.channels).reshape(1, -1, 1, 1)


if __name__ == "__main__":
    # input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                         [2.0, 3.0, 4.0, 5.0],
    #                         [3.0, 4.0, 5.0, 6.0],
    #                         [4.0, 5.0, 6.0, 7.0]]]])
    input = torch.rand((2, 2, 5, 5)) - 0.5
    print(input)
    
    batchnorm = BatchNorm(input.shape)
    
    out = batchnorm.forward(input)
    print(out)
    
    eta = batchnorm.gradient(out)
    print(eta)
    batchnorm.backward()