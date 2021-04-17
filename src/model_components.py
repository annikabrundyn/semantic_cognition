import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, img_size, hidden_size):
        super(Net, self).__init__()

        # TODO: make it possible to substitute representation layer (e.g. pretrained resnet, etc)
        self.representation_layer = SimpleCNN(in_channels=3)
        rep_size = self._calculate_rep_size(img_size)

        self.hidden_layer = nn.Linear(rep_size+4, hidden_size)
        self.attribute_layer = nn.Linear(hidden_size, 36)


    def _calculate_rep_size(self, img_size):
        x = torch.rand(1, 3, img_size, img_size)
        y = self.representation_layer(x)
        y = y.view(y.shape[0], -1)
        return y.shape[1]



    def forward(self, img, rel):

        rep = F.relu(self.representation_layer(img))
        rep = rep.view(rep.shape[0], -1)

        input_hidden = torch.cat([rep, rel], dim=1)
        hidden = F.relu(self.hidden_layer(input_hidden))

        output = torch.sigmoid(self.attribute_layer(hidden))
        return output, hidden, rep


class SimpleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        return x

