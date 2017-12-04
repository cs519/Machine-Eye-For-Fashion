import time
import copy
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from torch.autograd import Variable

from preprocessing.preprocessing import make_dsets, get_label_idx_to_name, image_loader, default_loader, get_transforms
from model import utils


class AttributeFC(nn.Module):
    """
    Attributes model with fully connected output layers
    """
    def __init__(self, pretrained_fc, fc_dim, output_shape):
        """
        Constructor for class

        :param pretrained_fc: Pre-trained model's fully connected layers
        :param fc_dim: Number of outputs of the pre-trained model's fully connected layers
        :param output_shape: Number of desired outputs for the model
        """
        super().__init__()

        # remove the last linear layer, and a replace with a a trainable Linear layer
        layers = list(
            pretrained_fc.children())[:-1] + [nn.Linear(fc_dim, output_shape)]
        # assign the model as an attribute to the class
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass for the model

        :param x: input to the network
        """
        # Set the input to the network
        # Use softmax on the last linear layer to give results
        return F.softmax(self.model(x))


class AttributeFCN(nn.Module):
    """
    Attributes model with fully convolutional output layers
    """
    def __init__(self, input_shape, output_shape, return_conv_layer=False):
        """
        Constructor for class

        :param 
        """
        super().__init__()
        self.return_conv_layer = return_conv_layer

        self.model = nn.Sequential(
            nn.BatchNorm2d(input_shape),
            nn.Conv2d(input_shape, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.Conv2d(64, output_shape, 1))

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


class AttributePredictDataset(data.Dataset):

    def __init__(self,
                 image_url,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        super().__init__()

        self.image_url = image_url
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, idex):
        image = self.loader(self.image_url)
        target = 0

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return self.image_url, image, target

    def __len__(self):
        return 1


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


def train_attribute_model(model,
                          pretrained_model,
                          train_dset_loader,
                          valid_dset_loader=None,
                          criterion=nn.CrossEntropyLoss(),
                          optim_scheduler=optim_scheduler,
                          use_gpu=None,
                          num_epochs=25,
                          verbose=False,
                          flatten_pretrained_out=False):

    since = time.time()
    best_model = model
    best_acc = 0.0

    if not use_gpu:
        use_gpu = torch.cuda.is_available()

    if use_gpu:
        pretrained_model.cuda()
        model.cuda()

    phases = ['train']
    dset_sizes = {'train': len(train_dset_loader.dataset)}

    if valid_dset_loader:
        phases.append('valid')
        dset_sizes['valid'] = len(valid_dset_loader.dataset)

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in phases:
            if phase == 'train':
                optimizer = optim_scheduler(model, epoch)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data
            dset_loader = valid_dset_loader if phase == 'valid' else train_dset_loader

            for data in dset_loader:
                batch_files, inputs, labels = data

                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                if phase == 'train':
                    inputs = Variable(inputs)
                else:
                    inputs = Variable(inputs, volatile=True)
                labels = Variable(labels)

                out_features = pretrained_model(inputs)
                if flatten_pretrained_out:
                    out_features = out_features.view(out_features.size(0), -1)

                # Forward
                outputs = model(out_features)
                loss = F.nll_loss(outputs, labels)
                preds = outputs.data.max(1)[1]

                # Backward + Optimize
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            elif epoch % 3 == 0:
                print("{} Epoch {}/{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch, num_epochs - 1, epoch_loss, epoch_acc))

            # Check of the model in the current stage produces the best resutls
            # If so, update bests variables
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('Training completed in {:0f}m and {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model


def train_model(model,
                pretrained_features,
                target_column,
                labels_file,
                train_images_folder,
                valid_images_folder=None,
                batch_size=32,
                num_workers=4,
                num_epochs=10,
                use_gpu=None,
                flatten_pretrained_out=False):

    # Make training and validation datasets for training
    train_dset_loader = make_dsets(
        train_images_folder,
        labels_file,
        target_column,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=False)

    # check if validation images exists
    # if yes, create valdation dataset
    valid_dset_loader = None
    if valid_images_folder:
        valid_dset_loader = make_dsets(
            valid_images_folder,
            labels_file,
            target_column,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=False)

    model = train_attribute_model(
        model,
        pretrained_features,
        train_dset_loader=train_dset_loader,
        valid_dset_loader=valid_dset_loader,
        num_epochs=num_epochs,
        use_gpu=use_gpu,
        flatten_pretrained_out=flatten_pretrained_out)

    return model


def predict_model(model, inputs, flatten=False):
    outputs = model(inputs)
    if flatten:
        outputs = outputs.view(outputs.size(0), -3)
    return outputs


def test_models(attribute_models,
                pretrained_model,
                image_url,
                attribute_idx_map=None,
                use_gpu=None,
                return_last_conv_layer=False):

    if not use_gpu:
        use_gpu = torch.cuda.is_available()

    image_dset = AttributePredictDataset(
        image_url, transform=get_transforms(is_train=False))

    print(image_dset)

    dset_loader = data.DataLoader(image_dset, shuffle=False)

    results = {}

    for attribute_name, model in attribute_models.items():
        model.eval()

        for batch_idx, dset, in enumerate(dset_loader):
            batch_files, inputs, labels = dset

            if use_gpu:
                inputs, labels, = inputs.cuda(), labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)
            out_features = pretrained_model(inputs)

            # Forward
            outputs = model(out_features)
            if return_last_conv_layer:
                conv_layer_out = model.conv_layer_out(out_features)

            loss = F.nll_loss(outputs, labels)
            preds_proba, preds = outputs.data.max(1)
            pred_idx = preds.cpu().numpy().flatten()[0]
            preds_proba = np.exp(preds_proba.cpu().numpy.flatten()[0])

            if attribute_idx_map:
                pred_class = attribute_idx_map[attribute_name].get(pred_idx)

                if pred_class:
                    results[attribute_name] = {
                        'pred_idx': pred_idx,
                        'pred_prob': preds_proba,
                        'pred_class': pred_class
                    }

                    if return_last_conv_layer:
                        results[attribute_name]['conv_layer'] = conv_layer_out

    return results

#Fuction to evaluate each model
def evaluate_model(model,
                   pretrained_model,
                   target_column,
                   labels_file,
                   image_folder,
                   batch_size=32,
                   num_workers=4,
                   use_gpu=None,
                   flatten_pretrained_out=False):

    #Set to use GPU
    if not use_gpu:
        use_gpu = torch.cuda.is_available()

    #perform preprocessing
    dset_loader = make_dsets(
        image_folder,
        labels_file,
        target_column,
        is_train=False,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers)

    running_loss = 0
    running_corrects = 0
    y_actual = []
    y_pred = []
    y_pred_prob = []
    input_files = []

    model.eval()

    print(dset_loader)

    for batch_idx, data in enumerate(dset_loader):
        batch_files, inputs, labels = data
        
        y_actual = np.concatenate([y_actual, labels.numpy()])
        input_files = np.concatenate([input_files, batch_files])

        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)
        out_features = pretrained_model(inputs)

        if flatten_pretrained_out:
            out_features = out_features.view(out_features.size(0, -1))

        # Forward
        outputs = model(out_features)
        loss = F.nll_loss(outputs, labels)
        preds_proba, preds = outputs.data.max(1)
        y_pred = np.concatenate([y_pred, preds.cpu().numpy().flatten()])
        y_pred_prob = np.concatenate(
            [y_pred_prob, preds_proba.cpu().numpy().flatten()])

        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    return {
        'loss': running_loss,
        'accuracy': running_corrects / len(y_actual),
        'y_actual': y_actual,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
    }


def predict_attributes(image_url,
                       pretrained_model,
                       attribute_models,
                       attribute_idx_map=None,
                       flatten_pretrained_out=True,
                       use_gpu=None):

    if not use_gpu:
        use_gpu = torch.cuda.is_available()

    image_features = image_loader(image_url, use_gpu=use_gpu)

    print(image_features.size())

    pretrained_features = predict_model(
        pretrained_model, image_features, flatten=flatten_pretrained_out)
    results = {}

    for attribute_name, model in attribute_models.items():
        print('Predicting {}'.format(attribute_name))

        outputs = predict_model(model, pretrained_features)
        pred_prob, pred_class = outputs.data.max(1)

        if use_gpu:
            pred_prob = pred_prob.cpu()
            pred_class = pred_class.cpu()

        pred_prob = np.exp(pred_prob.numpy().flatten()[0])
        pred_class = pred_class.numpy().flatten()[0]

        if attribute_idx_map:
            pred_class = attribute_idx_map[attribute_name].get(pred_class)
        if pred_class:
            results[attribute_name] = (pred_class, pred_prob)

    return results


def create_attributes_fcn_model(ModelClass,
                                input_shape,
                                pretrained_features,
                                target_columns,
                                weights_root,
                                labels_file,
                                train_images_folder,
                                valid_images_folder=None,
                                is_train=True,
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
        if is_train:
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
                flatten_pretrained_out=False)
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
                               is_train=True,
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

        if is_train:
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
                flatten_pretrained_out=True)
            # Save weights after completing training
            utils.save_model(model, weights_path)

        models[col_name] = model

    return models
