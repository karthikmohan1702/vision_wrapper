from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class cifar10_Net(nn.Module):
    def __init__(self, dropout_value = 0.06):
        super(cifar10_Net, self).__init__()

        # ---------- BLOCK - 1 -------------------
        self.conv1 = nn.Sequential(
            # Convolution Block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Transition block
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.Conv2d(32, 32, 3, stride=2, dilation=2),
        )

        # ---------- BLOCK - 2 - DILATION -------------------
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Transition Block
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.Conv2d(32, 32, 3, stride=2, dilation=2),
        )

        # ---------- BLOCK - 3 - DEPTHWISE-------------------
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, groups=32, padding=1),
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        # Transition Block
        self.trans3 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
        )

        # ---------- BLOCK - 4 -------------------
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.op = nn.Sequential(nn.AvgPool2d(3, 2), nn.Conv2d(64, 10, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)
        x = self.conv4(x)
        x = self.op(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)
