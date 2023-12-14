import torch

class SoftMax(object):
    def __init__(self, shape) -> None:
        self.softmax = torch.zeros(shape)
        self.eta = torch.zeros(shape)
        self.batch_size = shape[0]
    
    def predict(self, x):
        prediction = torch.zeros(x.shape)
        self.softmax = torch.zeros(x.shape)
        for b_idx in range(self.batch_size):
            x[b_idx, :] -= torch.max(x[b_idx, :])
            prediction[b_idx] = torch.exp(x[b_idx])
            self.softmax[b_idx] = prediction[b_idx] / torch.sum(prediction[b_idx])
        return self.softmax
    
    def calc_loss(self, x, label):
        self.label = label
        self.x = x
        self.predict(x)
        self.loss = 0
        for b_idx in range(self.batch_size):
            loss = torch.log(torch.sum(torch.exp(x[b_idx]))) - x[b_idx][label[b_idx].item()]
            self.loss += loss
        return self.loss
    
    def gradient(self):
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