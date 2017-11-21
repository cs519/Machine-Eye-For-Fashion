import torch
import torch.nn as nn
import torchvision


class FashionNet(nn.Module):
    """
    FashionNet model"""
    def __init__(self, vgg_features, pose_features):
        super().__init__()
        self.vgg_features = vgg_features
        self.pose_features = pose_features
        self.fc_pose = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1),
            nn.ReLU(True),
            nn.Linear(1, 1),
            nn.ReLU(True)
        )

        self.location = nn.Sequential(
            nn.Linear(1, 1)
        )

        self.visibility = nn.Sequential(
            nn.Linear(1, 1),
            nn.Softmax(),
            nn.Linear(1, 1),
            nn.Softmax(),
            nn.Linear(1, 1),
            nn.Softmax(),
            nn.Linear(1, 1),
            nn.Softmax()
        )


    def forward(self, x, y):
        x = self.vgg_features(x)
        x = self.pose_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_pose(x)
        x = self.location(x)
        y = self.visibility(x)

        return (x, y)



def features(batch_norm=False):
    # Get pretrained vgg model
    vgg = torchvision.models.vgg16(pretrained=True)
    # Select up to conv4 block
    layers = list(vgg.features.children())[:-7]

    return nn.Sequential(*layers)


def make_landmark_pose_layers(batch_norm=False):
    layers = []

    # Make final convolution layers.
    conv2d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    for i in range(3):    
        if batch_norm:
            layers.append(conv2d)
            layers.append(nn.BatchNorm2d(512))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(conv2d)
            layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)

def landmark_network(batch_norm=False):
    vgg = features(batch_norm=batch_norm)
    pose = make_landmark_pose_layers(batch_norm=batch_norm)
    model = FashionNet(vgg, pose)

    return model


if __name__ == '__main__':
    m = landmark_network()
    print(m)
