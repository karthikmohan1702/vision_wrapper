from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, norm=None, dropout_value=0.1, num_groups=None):
        super(Net, self).__init__()

        self.convblock1 = self.conv2d(
            in_channels=1,
            out_channels=12,
            kernel_size=(3, 3),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.convblock2 = self.conv2d(
            in_channels=12,
            out_channels=12,
            kernel_size=(3, 3),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock3 = self.conv2d(
            in_channels=12,
            out_channels=24,
            kernel_size=(3, 3),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.convblock4 = self.conv2d(
            in_channels=24,
            out_channels=12,
            kernel_size=(1, 1),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.convblock5 = self.conv2d(
            in_channels=12,
            out_channels=12,
            kernel_size=(3, 3),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.convblock6 = self.conv2d(
            in_channels=12,
            out_channels=12,
            kernel_size=(3, 3),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.convblock7 = self.conv2d(
            in_channels=12,
            out_channels=24,
            kernel_size=(3, 3),
            dropout=dropout_value,
            norm=norm,
            num_groups=num_groups,
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))

        self.convblock8 = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )

    def conv2d(self, in_channels, out_channels, kernel_size, dropout, norm, num_groups):

        if norm == "BN":
            conv_op = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif norm == "GN":
            conv_op = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=0, bias=False
                ),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif norm == "LN":
            # All the channels and 1 group is equivalent to 1 layer & all the channels
            # Had to adopt this way to make the code dynamic, as the output size of the
            # image varies
            conv_op = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=0, bias=False
                ),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            conv_op = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        return conv_op

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
