import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    This is a custom resnet model that has 3 layers with two
    skip connections.
    """

    def __init__(self):
        super(ResNet, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )

        # Layer L1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        # Residual R1
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )

        # Layer L2
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        # Layer L3
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        # Residual R2
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Prep layer
        x = self.prep(x)

        # Layer 1
        l1 = self.layer1(x)
        r1 = self.res1(l1)

        # Sum of layer1 & residue
        sum_l1_r1 = l1 + r1

        # Layer 2
        l2 = self.layer2(sum_l1_r1)

        # Layer 3
        l3 = self.layer3(l2)
        r2 = self.res2(l3)

        # Sum of layer3 & residue
        sum_l3_r2 = l3 + r2

        # Max pool
        x = self.pool(sum_l3_r2)

        x = x.view(-1, 512)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)
