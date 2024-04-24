import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):  # Do NOT change the signature of this function
        super(CNN, self).__init__()
        self.n = 4
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2
        '''
        we use bach norm to normelized the data 
        before going to the next step to exalerate the 
        learning and to avoid gradient exploding
        '''
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.n,kernel_size=self.kernel_size,padding=self.padding)
        self.bn1 = nn.BatchNorm2d(self.n)
        self.conv2 = nn.Conv2d(in_channels=self.n,out_channels=2*self.n,kernel_size=self.kernel_size,padding=self.padding)
        self.bn2 = nn.BatchNorm2d(2*self.n)
        self.conv3 = nn.Conv2d(in_channels=2*self.n,out_channels=4*self.n,kernel_size=self.kernel_size,padding=self.padding)
        self.bn3 = nn.BatchNorm2d(4*self.n)
        self.conv4 = nn.Conv2d(in_channels=4*self.n,out_channels=8*self.n,kernel_size=self.kernel_size,padding=self.padding)
        self.bn4 = nn.BatchNorm2d(8*self.n)
        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(8*self.n*14*28,100)
        self.bn_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,2)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width

          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        out = nn.functional.relu(self.bn1(self.conv1(inp)))
        out = self.pool(out)

        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)

        out = nn.functional.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)

        out = nn.functional.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)

        out = out.reshape(-1, 32*self.n*7*14)
        out = nn.functional.relu(self.bn_fc(self.fc1(out)))
        #out = self.dropout(out)
        out = self.fc2(out)

        return out


class CNNChannel(nn.Module):
    def __init__(self):
        super(CNNChannel, self).__init__()
        self.n = 4
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding)

        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=2 *self.n, kernel_size=self.kernel_size, padding=self.padding)

        self.conv3 = nn.Conv2d(in_channels=2 * self.n, out_channels=4 * self.n, kernel_size=self.kernel_size, padding=self.padding)

        self.conv4 = nn.Conv2d(in_channels=4 * self.n, out_channels=8 * self.n, kernel_size=self.kernel_size, padding=self.padding)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(8*self.n*14*14, 100)

        self.fc2 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(p=0.5)
    # TODO: complete this class
    def forward(self, inp):  # Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width

          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''



        out = inp.reshape(inp.size(0), 2*inp.size(1),inp.size(2)//2,inp.size(3))

        out = nn.functional.relu(self.conv1(out))
        out = self.pool(out)

        out = nn.functional.relu(self.conv2(out))
        out = self.pool(out)

        out = nn.functional.relu(self.conv3(out))
        out = self.pool(out)

        out = nn.functional.relu(self.conv4(out))
        out = self.pool(out)
        out = out.reshape(-1, 8*self.n*14*14)

        out = nn.functional.relu(self.fc1(out))
        #out = self.dropout(out)
        out = self.fc2(out)

        return out