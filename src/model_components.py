import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self,
                 feat_extractor: str,
                 crop_size: int,
                 rep_size: int,
                 hidden_size: int
                 ):
        super(Net, self).__init__()

        try:
            if feat_extractor == 'simple':
                self.representation_layer = SimpleCNN(in_channels=3, rep_size=rep_size)
            elif feat_extractor == 'resnet':
                self.representation_layer = PretrainedResnet18(rep_size=rep_size)
        except ValueError:
            print("No proper feature extractor name provided")

        self.hidden_layer = nn.Linear(rep_size+4, hidden_size)
        self.attribute_layer = nn.Linear(hidden_size, 36)

    # def _calculate_rep_size(self, img_size):
    #     x = torch.rand(1, 3, img_size, img_size)
    #     y = self.representation_layer(x)
    #
    #     self.rep3d_shape = y.squeeze(0).shape
    #
    #     y = y.view(y.shape[0], -1)
    #     return y.shape[1]

    def forward(self, img, rel):

        rep = F.relu(self.representation_layer(img))
        rep_flat = rep.view(rep.shape[0], -1)

        input_hidden = torch.cat([rep_flat, rel], dim=1)
        hidden = F.relu(self.hidden_layer(input_hidden))

        output = torch.sigmoid(self.attribute_layer(hidden))
        return output, hidden, rep



class SimpleCNN(nn.Module):
    def __init__(self, in_channels, rep_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        # TODO: this is still hard coded - relies on 64 crop size
        self.fc = nn.Linear(32*29*29, rep_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# NOTE - the image size must be greater than 64 x 64 for this to work from what i can tell (bc of the many downsampling layers)
class PretrainedResnet18(nn.Module):
    def __init__(self, rep_size):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()
        self.fc = nn.Linear(2048, rep_size)

        # freeze all layers
        #for param in self.resnet.parameters():
            #param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
