import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=0, stride=1)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=(5, 5), padding=0, stride=1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120,kernel_size = (5,5), padding=0,stride=1)
        self.L1 = nn.Linear(120, 84)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = nn.Tanh()
        self.representation_dim = 84

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.act(x)
        x = x.view(x.size()[0], -1)
        x = self.L1(x)

        return x

