import os
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


class AttributeFC(nn.Module):

    def __init__(self, pretrained_fc, fc_dim, output_shape):
        super().__init__()
        layers = list(
            pretrained_fc.children())[:-1] + [nn.Linear(fc_dim, output_shape)]

    def forward(self, x):
        return F.softmax(self.model(x))


class AttributeFCN(nn.Module):

    def __init__(self, input_shape, output_shape, return_conv_layer=False):
        super().__init__()
        self.return_conv_layer = return_conv_layer

        self.model = nn.Sequential([
            nn.BatchNorm2d(input_shape),
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
    # decrease learning rate every 7 epochs
    updated_lr = lr * (0.1**(epoch // decay))

    # Show current learning rate
    if epoch % decay == 0:
        print('LR is now {}'.format(updated_lr))

    # Use Stochastic Gradient Decent with a momentum of 0.9
    optimizer = optim.SGD(model.parameters(), lr=updated_lr, momentum=0.9)

    return optimizer


def precision(output, target, topk=(1,)):
    """Computes the precision for the top k results"""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


#TODO: train_attribute_model

#TODO: train_model


def predict_model(model, inputs, flatten=False):
    outputs = model(inputs)
    if flatten:
        outputs = outputs.view(outputs.size(0), -3)
    return outputs


#TODO: test_models

#TODO: evaluate_model

#TODO: predict_attributes


def create_attributes_fcn_model(ModelClass,
                                input_shape,
                                pretrained_features,
                                target_columns,
                                weights_root,
                                labels_file,
                                train_images_folder,
                                valid_images_folder=None,
                                mode='train',
                                batch_size=32,
                                num_workers=4,
                                num_epochs=10,
                                use_gpu=None):

    # Each target needs its own model
    models = {}

    # Create a model for each target attribute/label
    for col_name, col_shape in target_columns.items():
        print("Processing Attribute: {}".format(col_name))

        # Set path for weights file (weights may or may not exist)
        weights_path = os.path.join(weights_root, col_name + ".pth")
        load_weights_path = None

        # Only load weights if it exists already
        if os.path.exists(weights_path):
            load_weights_path = weights_path

        # Get new FCN model
        model = utils.load_fcn_model(
            ModelClass,
            input_shape,
            col_shape,
            weights_path=load_weights_path,
            return_conv_layer=False,
            use_gpu=use_gpu)

        # Decide it model will be used for training or testing
        if mode == 'train':
            print("Start training for: {}".format(col_name))
            # Train the model
            model = train_model(
                model,
                pretrained_features,
                col_name,
                labels_file,
                train_images_folder,
                valid_images_folder,
                batch_size,
                num_workers,
                num_epochs,
                use_gpu=use_gpu,
                latten_pretrained_out=False)
            # Save weights after completing training
            utils.save_model(model, weights_path)

        models[col_name] = model

    return models


def create_attributes_fc_model(ModelClass,
                               pretrained_fc,
                               pretrained_features,
                               fc_shape,
                               target_columns,
                               weights_root,
                               labels_file,
                               train_images_folder,
                               valid_images_folder=None,
                               mode='train',
                               batch_size=32,
                               num_workers=4,
                               num_epochs=10,
                               use_gpu=None):

    # Each target needs its own model
    models = {}

    # Create a model for each target attribute/label
    for col_name, col_shape in target_columns.items():
        print("Processing Attribute: {}".format(col_name))

        # Set path for weights file (weights may or may not exist)
        weights_path = os.path.join(weights_root, col_name + ".pth")
        load_weights_path = None

        # Only load weights if it exists already
        if os.path.exists(weights_path):
            load_weights_path = weights_path

        # Get new Dense model
        model = utils.load_fc_model(
            ModelClass,
            pretrained_fc,
            fc_shape,
            col_shape,
            weights_path=load_weights_path,
            use_gpu=use_gpu)

        if mode == 'train':
            print("Start training for: {}".format(col_name))
            # Train the model
            mode = train_model(
                model,
                pretrained_features,
                col_name,
                labels_file,
                train_images_folder,
                valid_images_folder,
                batch_size,
                num_workers,
                num_epochs,
                use_gpu=use_gpu,
                flatten_pretrained_out=True)
            # Save weights after completing training
            utils.save_model(model, weights_path)

        models[col_name] = model

    return models
