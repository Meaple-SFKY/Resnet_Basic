import torch
from torch import Tensor

class SoftMax(object):
    def __init__(self, shape, device='cpu') -> None:
        self.softmax = torch.zeros(shape, requires_grad=False).to(device)
        self.eta = torch.zeros(shape, requires_grad=False).to(device)
        self.batch_size = shape[0]
        self.device = device
        self.prediction = torch.zeros(shape, requires_grad=False).to(self.device)
        self.loss = torch.tensor(0.0, requires_grad=False).to(device)
    
    def predict(self, x) -> Tensor:
        torch.zero_(self.prediction)
        torch.zero_(self.softmax)
        for b_idx in range(self.batch_size):
            x[b_idx, :] -= torch.max(x[b_idx, :])
            self.prediction[b_idx] = torch.exp(x[b_idx])
            self.softmax[b_idx] = self.prediction[b_idx] / torch.sum(self.prediction[b_idx])
        return self.softmax
    
    def calc_loss(self, x, label) -> Tensor:
        torch.zero_(self.loss)
        self.label = label
        self.x = x
        self.predict(x)
        for b_idx in range(self.batch_size):
            loss = torch.log(torch.sum(torch.exp(x[b_idx]))) - x[b_idx][label[b_idx].item()]
            self.loss += loss
        return self.loss
    
    def gradient(self) -> Tensor:
        self.eta = self.softmax.clone()
        for b_idx in range(self.batch_size):
            self.eta[b_idx, self.label[b_idx]] -= 1
        return self.eta


if __name__ == '__main__':
    # input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                         [2.0, 3.0, 4.0, 5.0],
    #                         [3.0, 4.0, 5.0, 6.0],
    #                         [4.0, 5.0, 6.0, 7.0]]]])
    input = torch.rand((3, 10))

    print(input)
    sf = SoftMax(input.shape)
    out = sf.predict(input)
    print(input)