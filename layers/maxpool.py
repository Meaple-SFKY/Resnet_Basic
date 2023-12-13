import torch
import math

class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.batch_size = shape[0]
        self.out_channel = shape[1]
        self.max_index = torch.zeros(shape)
        self.pad_shape = (0, (shape[3] - ksize) % self.stride, 0, (shape[2] - ksize) % self.stride)
        self.pad = torch.nn.ZeroPad2d(padding=self.pad_shape)
        self.out_shape = [self.batch_size, self.out_channel, math.floor((shape[2] - ksize) / self.stride) + 1, math.floor((shape[3] - ksize) / self.stride) + 1]
    
    def forward(self, x):
        out = torch.zeros(self.out_shape)
        print(out.shape)
        
        for b_idx in range(self.batch_size):
            for c_idx in range(self.out_channel):
                for h_idx in range(self.out_shape[2]):
                    for w_idx in range(self.out_shape[3]):
                        start_h = h_idx * self.stride
                        start_w = w_idx * self.stride
                        end_h = start_h + self.ksize
                        end_w = start_w + self.ksize
                        out[b_idx, c_idx, h_idx, w_idx] = torch.max(x[b_idx, c_idx, start_h:end_h, start_w:end_w])
                        max_index = torch.argmax(x[b_idx, c_idx, start_h:end_h, start_w:end_w])
                        self.max_index[b_idx, c_idx, math.floor(start_h + max_index / self.ksize), math.floor(start_w + max_index % self.ksize)] = torch.tensor(1)
        return out
    
    def gradient(self, eta):
        mid_out = self.pad(torch.repeat_interleave(torch.repeat_interleave(eta, self.stride, 2), self.stride, 3))
        return mid_out * self.max_index

if __name__ == "__main__":
    # input = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
    #                         [2.0, 3.0, 4.0, 5.0],
    #                         [3.0, 4.0, 5.0, 6.0],
    #                         [4.0, 5.0, 6.0, 7.0]]]])
    input = torch.rand((2, 2, 5, 5))

    pool = MaxPooling(input.shape, 2, 2)
    out = pool.forward(input)
    print(out)
    grad = pool.gradient(out)
    print(grad)