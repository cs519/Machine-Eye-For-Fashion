import torch
import torch.nn as nn
import torchvision


class FashionNet(nn.Module):
    """
    Modified FashionNet: use model to find the landmarks of clothing
    """
    def __init__(self, vgg_features, pose_features, num_outputs):
        """
        Constructor for FashionNet class

        :param vgg_features: Feature layers of VGG-16
        :param pose_features: Layers for pose
        :param num_outputs: Number of landmarks
        """
        super().__init__()
        # Multiply by 2 because each landmark needs xy coordinate
        outputs = num_outputs * 2
        # each landmark has one of three visibilities: visible, partially visible, not visible
        classes = num_outputs * 3

        # Create the model
        # Save pretrained model
        self.vgg_features = vgg_features
        # Save the layers for pose
        self.pose_features = pose_features
        # Create layers for fully connected pose layers
        self.fc_pose = nn.Sequential(
            # Linear layer. Inputs = 25,088, Outputs = 8
            nn.Linear(512 * 7 * 7, 1),
            # ReLU activation layer
            nn.ReLU(inplace=True),
            # Linear Layer. Inputs = 1, Outputs = 1
            nn.Linear(1, 1),
            # ReLU activation layer
            nn.ReLU(inplace=True)
        )

        # Output layer for landmark locations
        self.location = nn.Sequential(
            # Linear layer. Inputs: 1, Outputs: num_outputs * 2
            nn.Linear(1, outputs)
        )

        # Output layer for landmark visibility
        self.visibility = nn.Sequential(
            # Linear Layer. Inputs (Landmark locations): num_outputs * 2, Outputs: num_outputs * 3
            nn.Linear(outputs, classes),
            # Softmax activation
            nn.Softmax(),
            # Linear Layer. Inputs = num_outputs * 3, Outputs = num_outputs * 3
            nn.Linear(classes, classes),
            # Softmax activation
            nn.Softmax(),
            # Linear Layer. Inputs = num_outputs * 3, Outputs = num_outputs * 3
            nn.Linear(classes, classes),
            # Softmax activation
            nn.Softmax(),
            # Linear Layer. Inputs = num_outputs * 3, Outputs = num_outputs * 3
            nn.Linear(classes, classes),
            # Softmax activation
            nn.Softmax()
        )

    # Forward pass through network
    def forward(self, x):
        # Send inputs through VGG
        x = self.vgg_features(x)
        # Send result from VGG to pose_features layers
        x = self.pose_features(x)
        # flatten
        x = x.view(x.size(0), - 1)
        # Send flattened to fc_pose layers
        x = self.fc_pose(x)
        # get the landmarks
        landmarks = self.location(x)
        # get visibility of landmarks
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
    """Create a new instace of FashionNet
    
    :param batch_norm: A boolean deciding if batch normalization layers
    :param num_outputs: A integer for the number of landmarks"""

    # Get pretrained VGG-16
    vgg = features(batch_norm=batch_norm)

    # set VGG-16 weights to not be updated
    for param in vgg.parameters():
        param.requires_grad = False

    # Make the pose layers
    pose = make_landmark_pose_layers(batch_norm=batch_norm)
    # Create a new instace of FashionNet
    model = FashionNet(vgg, pose, num_outputs)

    # Check if cuda is available, if so, use it.
    if torch.cuda.is_available():
        model = model.cuda()

    return model
