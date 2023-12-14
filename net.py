import torch
import torch.nn as nn

from layers.relu import Relu
from layers.batchnorm import BatchNorm
from layers.maxpool import MaxPooling
from layers.conv import Conv2D
from layers.linear import Linear
from layers.softmax import SoftMax

import numpy as np
import struct
from tqdm import tqdm

from glob import glob

data_path = r'/Users/sfky/Projects/datasets/Mnist'

def load_mnist(path, mode='train'):
    images_path = glob('%s/%s*3-ubyte' % (path, mode))[0]
    labels_path = glob('%s/%s*1-ubyte' % (path, mode))[0]
    
    with open(labels_path, 'rb') as label_file:
        _ = struct.unpack('>II', label_file.read(8))
        labels = torch.from_numpy(np.fromfile(label_file, dtype=np.uint8)).type(torch.int8)
    
    with open(images_path, 'rb') as image_file:
        _ = struct.unpack('>IIII', image_file.read(16))
        images = torch.from_numpy(np.fromfile(image_file, dtype=np.uint8).reshape(len(labels), 784)).type(torch.float32)
    
    return images, labels

images, labels = load_mnist(data_path, 'train')
test_imgs, test_labels = load_mnist(data_path, 't10k')
train_cnt = len(images)
test_cnt = len(test_imgs)
print('train count: ', train_cnt)
print(' test count: ', test_cnt)

batch_size = 40

input = torch.rand((batch_size, 1, 28, 28))

class Net(object):
    def __init__(self) -> None:
        self.conv1 = Conv2D(in_channels=1, out_channels=8, kernel_size=5, input_shape=28, stride=1, padding=0, batchsize=batch_size, bias=True, const_ker=False)
        self.pool1 = MaxPooling((batch_size, 8, 24, 24), 2, 2)
        self.relu1 = Relu((batch_size, 8, 12, 12))

        self.conv2 = Conv2D(in_channels=8, out_channels=32, kernel_size=3, input_shape=12, stride=1, padding=0, batchsize=batch_size, bias=True, const_ker=False)
        self.pool2 = MaxPooling((batch_size, 32, 10, 10), 2, 2)
        self.relu2 = Relu((batch_size, 32, 5, 5))

        self.conv3 = Conv2D(in_channels=32, out_channels=32, kernel_size=2, input_shape=5, stride=1, padding=0, batchsize=batch_size, bias=True, const_ker=False)
        self.pool3 = MaxPooling((batch_size, 32, 4, 4), 2, 2)
        self.relu3 = Relu((batch_size, 32, 2, 2))

        self.linear1 = Linear((batch_size, 32, 2, 2), 128, 32, True)
        self.linear2 = Linear((batch_size, 32), 32, 10, True)

    def forward(self, x):
        out_1 = self.relu1.forward(self.pool1.forward(self.conv1.forward(x)))
        out_2 = self.relu2.forward(self.pool2.forward(self.conv2.forward(out_1)))
        out_3 = self.relu3.forward(self.pool3.forward(self.conv3.forward(out_2)))
        out_4 = self.linear1.forward(out_3)
        out_5 = self.linear2.forward(out_4)
        return out_5
    
    def backward(self, loss):
        eta_5 = self.linear2.gradient(loss)
        eta_4 = self.linear1.gradient(eta_5)
        eta_3 = self.conv3.gradient(self.pool3.gradient(self.relu3.gradient(eta_4)))
        eta_2 = self.conv2.gradient(self.pool2.gradient(self.relu2.gradient(eta_3)))
        eta_1 = self.conv1.gradient(self.pool1.gradient(self.relu1.gradient(eta_2)))
        
        self.linear2.backward()
        self.linear1.backward()
        self.conv3.backward()
        self.conv2.backward()
        self.conv1.backward()

net = Net()
softmax = SoftMax((batch_size, 10))

for epoch in range(20):
    lr = 0.00001
    
    batch_loss = 0
    batch_acc = 0
    val_loss = 0
    val_acc = 0
    
    train_loss = 0
    train_acc = 0
    
    for i in tqdm(range(int(train_cnt / batch_size))):
        image = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 1, 28, 28])
        label = labels[i * batch_size:(i + 1) * batch_size]
        out = net.forward(image)
        loss = softmax.calc_loss(out, label)
        batch_loss += loss
        train_loss += loss
        
        for j in range(batch_size):
            if torch.argmax(softmax.softmax[j]) == label[j]:
                batch_acc += 1
                train_acc += 1
        softmax.gradient()
        net.backward(softmax.eta)
        
        if i % 10 == 0:
            print("epoch: %d, batch: %5d, avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch, i, batch_acc / float(batch_size), batch_loss / batch_size, lr))
        
        batch_loss = 0
        batch_acc = 0
    print("epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))