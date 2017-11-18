import torch
import torch.nn as nn
import torchvision


class FashionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.layers = list(self.vgg.features.children())
        print('type = {}, {}'.format(type(self.layers), self.layers))

    def forward(self, x):
        pass

    def backward(self, x):
        pass


if __name__ == '__main__':
    FashionNet()
