import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 
from torch.autograd import Variable


class AttributeFC(nn.Module):
    def __init__(self, pretrained_fc, fc_dim, output_shape):
        super().__init__()
        layers = list(pretrained_fc.children())[:-1] + [nn.Linear(fc_dim, output_shape)]

    def forward(self, x):
        return F.softmax(self.model(x))

class AttributeFCN(nn.Module):
    def __init__(self, input_shape, output_shape, return_conv_layer=False):
        super().__init__()
        self.return_conv_layer = return_conv_layer

        self.model = nn.Sequential([
            nn.Conv2d(input_shape, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, output_shape, 1, stride=1, padding=0)
        ])

    def forward(self, x):
        classes_conv_out = self.model(x)

        pool_size = (classes_conv_out.size(2), classes_conv_out.size(3))
        pool = F.avg_pool2d(classes_conv_out, kernel_size=pool_size)
        flatten = pool.view(pool.size(0), -1)
        classes = F.log_softmax(flatten)

        if self.return_conv_layer:
            return classes_conv_out, classes
        else:
            return classes

def optim_scheduler(model, epoch, lr=0.01, decay=7):
    updated_lr = lr * (0.1 ** (epoch // decay))

    # Show current learning rate
    if epoch % decay == 0:
        print('LR is now {}'.format(updated_lr))

    optimizer = optim.SGD(model.parameters(), lr=updated_lr, momentum=0.9)

    return optimizer

#TODO: accuracy

#TODO: train_attribute_model

#TODO: train_model

#TODO: predict_model
def predict_model(model, inputs, flatten=False):
    outputs = model(inputs)
    if flatten:
        outputs = outputs.view(outputs.size(0), -3)
    return outputs

#TODO: test_models

#TODO: evaluate_model

#TODO: predict_attributes

#TODO: create_attributes_model

#TODO: create_attributes_fc_model