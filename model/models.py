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
        """Define the forward pass for the model

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

        :param input_shape: Number of inputs to the fine-tuned section of the model. This number should be the output of the feature extractor.
        :param output_shape: Number of desired outputs for the model
        :param return_conv_layer: Boolean on whether to return the state of the last convolutional layer; default: False
        """
        super().__init__()
        # Save caller's choice this value to use later
        self.return_conv_layer = return_conv_layer

        # Create the fine-tuned layers
        self.model = nn.Sequential(
            # Normalize the outputs of the feature extractor
            nn.BatchNorm2d(input_shape),
            # 2D Convolution with number inputs equal to output of feature of extractor (input_shape), outputs = 256, kernel size = 3
            nn.Conv2d(input_shape, 256, 3, stride=1, padding=1),
            # Activation layer with ReLU
            nn.ReLU(),
            # Normalize with 2D BatchNormalizaton
            nn.BatchNorm2d(256),
            # 2D Convolution with inputs = 156, outputs = 128, kernel_size = 3
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            # Activation layer with ReLU
            nn.ReLU(),
            # Normalize with 2D BatchNormalizaton
            nn.BatchNorm2d(128),
            # 2D Convolution with inputs = 128, outputs = 63, kernel_size = 3
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # Activation layer with ReLU
            nn.ReLU(), 
            # Normalize with 2D BatchNormalizaton
            nn.BatchNorm2d(64), 
            # 2D Convolution with inputs = 63, outputs = output_shape, kernel_size = 1
            nn.Conv2d(64, output_shape, 1))

    def forward(self, x):
        """Define the forward pass for the model

        :param x: input to the network
        """

        # send input through model
        # save the results before softmax
        classes_conv_out = self.model(x)

        # define the size of the pooling layer's kernel_size
        pool_size = (classes_conv_out.size(2), classes_conv_out.size(3))
        # Do average pooling on results from sending input through network
        pool = F.avg_pool2d(classes_conv_out, kernel_size=pool_size)
        # flatten from 2d to 1d
        flatten = pool.view(pool.size(0), -1)
        # Do softmax to get the labels
        classes = F.log_softmax(flatten)

        # if caller wants the results before softmax, return it
        # this is used for visualizing how model thinks it's learning
        if self.return_conv_layer:
            return classes_conv_out, classes
        else:
            # return the classes model predicts
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

    """
    Train the model

    :param model: Model to train
    :param pretrained_model: Pre-trained model to use
    :param train_dset_loader: Dataset loader for loading training data
    :param valid_dset_loader: Dataset loader for loading validation data
    :param criterion: loss function to use; Default: CrossEntropy
    :param optim_scheduler: Optimizer scheduler
    :param use_gpu: Boolean on whether or not to use GPU; Default: None
    :param num_epoch: Number of epochs to train data; Default: 25
    :param verbose: Boolean for verbose level; Default: False
    :param flatten_pretrained_out: Boolean on whether or not flatten feature extractor
    """

    # Get the start time of training
    since = time.time()

    # Save the current state of the model, current best is 0
    best_model = model
    best_acc = 0.0

    if not use_gpu:
        use_gpu = torch.cuda.is_available()

    # Check if there is GPU available, if so, use it
    if use_gpu:
        pretrained_model.cuda()
        model.cuda()

    # add a train phase
    phases = ['train']
    # set the size of the training dataset
    dset_sizes = {'train': len(train_dset_loader.dataset)}

    # if there is a validation dataset loader, update phase and dataset sizes
    if valid_dset_loader:
        # append validation phase to phases
        phases.append('valid')
        # append size of validation set to dataset dictionary
        dset_sizes['valid'] = len(valid_dset_loader.dataset)

    # Train the model for the number of specified epochs
    for epoch in range(num_epochs):
        # Print out the current epoch if verbose is set to true
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # complete the training or validation step
        for phase in phases:
            # if train phase set the an optimizer, else eval the model
            if phase == 'train':
                optimizer = optim_scheduler(model, epoch)
            else:
                model.eval()

            # initialize the running loss and correct for the epoch
            running_loss = 0.0
            running_corrects = 0.0

            # Set the correct data_loader for the phase
            dset_loader = valid_dset_loader if phase == 'valid' else train_dset_loader

            # Iterate over the data
            for data in dset_loader:
                # unpack the loaded data
                batch_files, inputs, labels = data

                # Wrap data in Variables
                if use_gpu:
                    # load onto GPU if available
                    inputs, labels = inputs.cuda(), labels.cuda()
                if phase == 'train':
                    inputs = Variable(inputs)
                else:
                    inputs = Variable(inputs, volatile=True)
                labels = Variable(labels)

                # send input through pretrained model
                out_features = pretrained_model(inputs)
                # flattent the returned features if true
                if flatten_pretrained_out:
                    out_features = out_features.view(out_features.size(0), -1)

                # Forward pass
                outputs = model(out_features)
                # calculate loss
                loss = F.nll_loss(outputs, labels)
                # get the predictions
                preds = outputs.data.max(1)[1]

                # Backward pass + Optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # sum the loss for the current epoch
                running_loss += loss.data[0]
                # sum the number of correcly classified inputs
                running_corrects += torch.sum(preds == labels.data)

            # Calculate the loss and accuracy for the epoch
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            # if verbose set, print the loss for each epoch, else print every 3 epoch
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

    # print the time it took to train
    time_elapsed = time.time() - since
    print('Training completed in {:0f}m and {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print the best accurcy for the model
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

    """
    Train the model

    :param model: Model to train
    :param pretrained_features: Pre-trained model's feature extractor
    :param target_column: Columns of attributes to train on
    :param labels_files: Path to labels file
    :param train_images_folder: Path to training image folder
    :param valid_image_folder: Path to validation image folder; Default: None
    :param batch_sizes: Integer for batch size; Default: 32
    :param num_workers: Number of workers to load data; Default: 4
    :param num_epoch: Number of epochs to train data; Default: 25
    :param use_gpu: Boolean on whether or not to use GPU; Default: None
    :param flatten_pretrained_out: Boolean on whether or not flatten feature extractor; Default: False
    """

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

    # Create a new attribute model
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
    """
    Predict the attributes for the inputs

    :param model: Model to use to predict attributes
    :param inputs: Images to predict attributes
    :param flatten: Boolean on whether or not to flatten the last layer of network in case it is not Linear; Default: False
    """
    # predict the results for the inputs
    outputs = model(inputs)
    # flatten if caller wants to do it
    if flatten:
        outputs = outputs.view(outputs.size(0), -3)

    # return the predicted attributes
    return outputs

class AttributePredictDataset(data.Dataset):
    """
    Dataloader for testing
    """
    def __init__(self, image_url, transform=None, target_transform=None, loader=default_loader):
        """
        Contstructor for Class

        :param image_url: Path to image
        :param transform: Transform on input image. Any neccessary preprocessing is done here; Default: None
        :param target_transform: Transform on target labels. Any neccessay preprocessing on labels done here; Default: None
        :param loader: Image loader; Default: Pillow imread
        """
        
        super().__init__()
        # Save all arguments as fields in class
        self.image_url = image_url
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Load the image. Method not called directly.

        :param index: Current index of image to load (only loading one image, so not used. __getitem__ required it though)
        """
        # load the image
        image = self.loader(self.image_url)
        # set default target to 0
        target = 0

        # preprocess the image if transform is given
        if self.transform:
            image = self.transform(image)
        # load the image's real target if transform is given
        if self.target_transform:
            target = self.target_transform(target)

        # return image path, image, image's target label
        return self.image_url, image, target

    def __len__(self):
        """
        Return the number of elements in the data loader
        """
        # only loading 1 image
        return 1


def test_models(attribute_models,
                pretrained_model,
                image_url,
                attribute_idx_map=None,
                use_gpu=None,
                return_last_conv_layer=False):

    """
    Predict the attributes for an image

    :param attribute_model: attribute model
    :param pretrained_model: Pre-trained feature extractor
    :param image_url: Path to test image
    :param attribute_idx_map: Attributes map; Default: None 
    :param use_gpu: Boolean on whether to use GPU; Default: None
    :param return_last_conv_layer: Boolean to return last before softmax; Default: False
    """

    # if use_gpu is not set, set it programmically
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    # load the image and it's label
    image_dset = AttributePredictDataset(
        image_url, transform=get_transforms(is_train=False))

    print(image_dset)

    # Load the image with a dataset loader and perform any necessary transform
    dset_loader = data.DataLoader(image_dset, shuffle=False)

    # create empty dictionary to store atributes
    results = {}

    # predict the result for each atrribute
    # each attribute needs its own model
    for attribute_name, model in attribute_models.items():
        # set model in evaluation mode
        model.eval()

        # load the data from dataset loader and unpack data
        for batch_idx, dset, in enumerate(dset_loader):
            batch_files, inputs, labels = dset

            # if using gpu, put on GPU
            if use_gpu:
                inputs, labels, = inputs.cuda(), labels.cuda()
            # Wrap data into variables
            inputs = Variable(inputs)
            labels = Variable(labels)

            # set input through pretrained network
            out_features = pretrained_model(inputs)

            # Forward
            outputs = model(out_features)
            if return_last_conv_layer:
                conv_layer_out = model.conv_layer_out(out_features)

            # calculate loss
            loss = F.nll_loss(outputs, labels)
            
            # get the predictions
            preds_proba, preds = outputs.data.max(1)
            # make sure preductions are not longer on GPU
            pred_idx = preds.cpu().numpy().flatten()[0]
            preds_proba = np.exp(preds_proba.cpu().numpy.flatten()[0])
            
            # map the predicted class from a number to the actualy tname
            if attribute_idx_map:
                pred_class = attribute_idx_map[attribute_name].get(pred_idx)

                if pred_class:
                    results[attribute_name] = {
                        'pred_idx': pred_idx,
                        'pred_prob': preds_proba,
                        'pred_class': pred_class
                    }

                    # return the convolution layer before softmax
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
    """
    Evaluate the model with a test dataset

    :param model: Model to evaluate
    :param pretrained_model: Pre-trained model to use
    :param target_column: Attribute to evaluate
    :param labels_file: Path to labels file
    :param image_folder: Path to images
    :param batch_size: Number of batches for test
    :param num_workers: Number of workers to load data
    :param use_gpu: Boolean on whether or not to use GPU
    :param flatten_pretrained_out: Boolean on whether or not to flatten the result of pre-trained model
    """

    #Set to use GPU
    if use_gpu is None:
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

    # initialize local varables
    running_loss = 0
    running_corrects = 0
    y_actual = []
    y_pred = []
    y_pred_prob = []
    input_files = []

    # set model in evalution mode
    model.eval()

    print(dset_loader)

    # load the data for data loader
    for batch_idx, data in enumerate(dset_loader):
        # unpack the data
        batch_files, inputs, labels = data
        
        # get the real targets
        y_actual = np.concatenate([y_actual, labels.numpy()])
        # get the input files
        input_files = np.concatenate([input_files, batch_files])

        # if using gpu, put data on GPU
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Wrap data with variables
        inputs = Variable(inputs)
        labels = Variable(labels)

        # send input through pre-trained model
        out_features = pretrained_model(inputs)

        # if true, flatten the last layer of the pre-trained network
        if flatten_pretrained_out:
            out_features = out_features.view(out_features.size(0, -1))

        # Forward
        outputs = model(out_features)
        # Calculate loss
        loss = F.nll_loss(outputs, labels)
        
        # unpack the prediction
        preds_proba, preds = outputs.data.max(1)
        # put the prediction on CPU
        y_pred = np.concatenate([y_pred, preds.cpu().numpy().flatten()])
        y_pred_prob = np.concatenate(
            [y_pred_prob, preds_proba.cpu().numpy().flatten()])

        # calculate the loss and correctly labeled images
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

    """
    :param image_url: Path to image to test
    :param pretrained_model: Pretrained model to use
    :param attribute_models: Attribute model to use
    :param attribute_idx_map: Attributes map; Default: None 
    :param flatten_pretrained_out: Boolean to flatten the outputs of pre-trained model; Default: True
    :param use_gpu: Boolean to use GPU; Default: None
    """

    # If use_gpu is not set, set it here programically
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    # Load the image
    image_features = image_loader(image_url, use_gpu=use_gpu)

    print(image_features.size())

    # predict the results with the pretrained model
    pretrained_features = predict_model(
        pretrained_model, image_features, flatten=flatten_pretrained_out)
    results = {}

    # predict the result for each atrribute
    # each attribute needs its own model
    for attribute_name, model in attribute_models.items():
        # print the attribute currently predicting
        print('Predicting {}'.format(attribute_name))

        # set the output of pre-trained model to attributes, model to get attributes
        outputs = predict_model(model, pretrained_features)
        # unpack the prediction
        pred_prob, pred_class = outputs.data.max(1)

        # if GPU was used, put predictions back on CPU
        if use_gpu:
            pred_prob = pred_prob.cpu()
            pred_class = pred_class.cpu()

        # flatten the prediction
        pred_prob = np.exp(pred_prob.numpy().flatten()[0])
        pred_class = pred_class.numpy().flatten()[0]

        # Map the pred to the class from the attributes file
        if attribute_idx_map:
            pred_class = attribute_idx_map[attribute_name].get(pred_class)
        # store the predictions
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

    """
    :param ModelClass: Class of Model to create
    :param input_shape: Shape of input to the first layer of network
    :param pretrained_features: Pre-trained feature extractor
    :param target_columns: Attributes to train
    :param weights_root: Path for weights folder
    :param labels_file: Path tot labels_files
    :param train_image_folder: Path to train images
    :param valid_image_folder: Path to validation images
    :param is_train: Boolean to train or test
    :param batch_size: Number for batch size
    :param num_workers: Number of workers to load data
    :param num_epochs: Number of epochs to train
    :param use_gpu: Boolean to use GPU
    """

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

        # Decide if model will be used for training or testing
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
        # Save the each model after training
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

    """
    :param ModelClass: Class of Model to create
    :param pretrained_fc: Pre-trained model's fully connected layers
    :param pretrained_features: Pre-trained feature extractor
    :param fc_shape: Number of outputs of feature extractor; used as number of inputs for next layer 
    :param target_columns: Attributes to train
    :param weights_root: Path for weights folder
    :param labels_file: Path tot labels_files
    :param train_image_folder: Path to train images
    :param valid_image_folder: Path to validation images
    :param is_train: Boolean to train or test
    :param batch_size: Number for batch size
    :param num_workers: Number of workers to load data
    :param num_epochs: Number of epochs to train
    :param use_gpu: Boolean to use GPU
    """

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

        # Decide if model will be used for training or testing
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
        # Save the each model after training
        models[col_name] = model

    return models
