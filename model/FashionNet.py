import torch
import torch.nn as nn
import torchvision


class FashionNet(nn.Module):
    """
    FashionNet model"""
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.layers = list(self.vgg.features.children())
        print('type = {}, {}'.format(type(self.layers), self.layers))

    def forward(self, x):
        pass

    def get_loss(l, v, f, n_int):
        t1 = 2000
        t2 = 400
        
        if n_int < t1 :
            pass #alpha = alpha // same with betha
        elif t1 < n_int < t2:
            pass #alpha = alpha*(t-t1)/(t2-t1) // samewith betha
        else:
            alpha = 0
            betha = 0

        #Calculate Euclidient Loss for Visibility, Landmark Positions and Labels

if __name__ == '__main__':
    FashionNet()
