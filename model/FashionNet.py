import torch
import torch.nn as nn
import torchvision


class FashionNet(nn.Module):
    """
    FashionNet model"""

    def __init__(self, vgg_features, pose_features, num_outputs):
        super().__init__()
        outputs = num_outputs * 2
        classes = num_outputs * 3
        self.vgg_features = vgg_features
        self.pose_features = pose_features
        self.fc_pose = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1), nn.ReLU(True), nn.Linear(1, 1),
            nn.ReLU(True))

        self.location = nn.Sequential(nn.Linear(1, outputs))

        self.visibility = nn.Sequential(
            nn.Linear(outputs, classes), nn.Softmax(),
            nn.Linear(classes, classes), nn.Softmax(),
            nn.Linear(classes, classes), nn.Softmax(),
            nn.Linear(classes, classes), nn.Softmax())

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.pose_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_pose(x)
        landmarks = self.location(x)
        vis = self.visibility(landmarks)

        return landmarks, vis


def features(batch_norm=False):
    # Get pretrained vgg model
    if batch_norm:
        vgg = torchvision.models.vgg16_bn(pretrained=True)
    else:
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


def landmark_network(batch_norm=False, num_outputs=8):
    vgg = features(batch_norm=batch_norm)

    for param in vgg.parameters():
        param.requires_grad = False

    pose = make_landmark_pose_layers(batch_norm=batch_norm)
    model = FashionNet(vgg, pose, num_outputs)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


if __name__ == '__main__':
    m = landmark_network(num_outputs=6)
    print(m)
