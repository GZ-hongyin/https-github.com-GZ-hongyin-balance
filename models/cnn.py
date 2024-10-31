import torch
import torch.nn as nn
import torch.nn.functional as F

# (1)LeNet
class LeNet_mnist(nn.Module):
    def __init__(self, n_in=1, n_out=10, size=28):         # channel number 
        super(LeNet_mnist, self).__init__()
        if size == 28:
            firstkernel_size = 3
        elif size == 32:
            firstkernel_size = 5
        else:
            raise ValueError("Input image size  was not  %d or %d" % (28, 32))

        self.conv1 = nn.Conv2d(n_in, 6, firstkernel_size, stride=1, padding=1)  #28->28
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0) #
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32 -> 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14-> 10 -> 5
        x = x.view(-1, self.num_flat_features(x))   # [-1, 400]为 [mini_batch, 400]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze()
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# (2)AlexNet
# ALexNet_cifar10：
class AlexNet(nn.Module):
    def __init__(self, n_out=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   #32-16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),    
            nn.Linear(4096, n_out),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
