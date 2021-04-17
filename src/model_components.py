import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, rep_size, hidden_size):
        super(Net, self).__init__()

        self.representation_layer = nn.Conv2d(3, 1, kernel_size=3, stride=1)
        ## calculate conv layer output size repszie =
        self.hidden_layer = nn.Linear(rep_size+4, hidden_size)
        self.attribute_layer = nn.Linear(hidden_size, 36)

    def forward(self, img, rel):

        rep = F.relu(self.representation_layer(img))
        rep = rep.view(rep.shape[0], -1)

        input_hidden = torch.cat([rep, rel], dim=1)
        hidden = F.relu(self.hidden_layer(input_hidden))

        output = F.sigmoid(self.attribute_layer(hidden))
        return output, hidden, rep


