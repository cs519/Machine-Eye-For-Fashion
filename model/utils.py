import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from matplotlib import pyplot as plt 
from matplotlib.pyplot import imshow


def get_pretrained_model(model='resnet18',
                         pop_last_pool_layer=False,
                         use_gpu=False):
    """Get pretrained model and return model's features, dense layer, and dense layer dimensions

    :param  model: A string of the model's name
    :param pop_last_pool_layer: A boolean
    :rtype pretrained_features: A torch Sequential container with the model's feature extracter
    :rtype pretrained_fc: A torch Linear container with the model's fully connected layers at the bottom of the network
    :rtype pretrained_features: An integer of Linear/Dense layer outputs 
    """
    # Check which model the caller wants, and load it from torchvision
    if model == 'resnet18':
        resnet = torchvision.models.resnet18(pretrained=True)
        pretrained_features = nn.Sequential(*list(resnet.children())[:-1])
        pretrained_fc = resnet.fc
        # Dimension of Dense/Fully Connected Layer
        fc_dim = 512
    elif model == 'vgg16':
        vgg = torchvision.models.vgg16(pretrained=True)
        pretrained_features = vgg.features
        pretrained_fc = vgg.classifier
        fc_dim = 4096
    elif model == 'densenet':
        densenet = torchvision.models.densenet121(pretrained=True)
        pretrained_features = nn.Sequential(*list(model.children())[:-1])
        pretrained_fc = densenet.classifier
        fc_dim = 1024
    # If the calleer wants to use the GPU, put the model on the GPU
    if use_gpu:
        pretrained_features.cuda()

    # remove the last pooling layer, used if want to fine tune the last pooling layer
    if pop_last_pool_layer:
        pretrained_features = nn.Sequential(*list(
            pretrained_features.children())[:-1])

    # Don't update the weights on the pretained features
    for param in pretrained_features.parameters():
        param.requires_grad = False

    return pretrained_features, pretrained_fc, fc_dim


def load_fcn_model(ModelClass,
                   input_shape,
                   output_shape,
                   weights_path=None,
                   return_conv_layer=False,
                   use_gpu=False):
    """
    :param ModelClass: Class for model to use
    :param input_shape: Number of outputs of feature extractor; used as number of inputs for next layer
    :param output_shape: Number of desired outputs for the network
    :param weights_path: Path to weights file; default: None
    :param return_conv_layer: Boolean on whether to return the state of the last convolutional layer; default: False
    :param use_gpu: Boolean on whether to use GPU; default: False
    """
    # Contruct a new model based on the given inputs
    model = ModelClass(input_shape, output_shape, return_conv_layer)

    # check if caller wants to use the GPU, place it there
    if use_gpu:
        model.cuda()

    # load the weights for the model's weights if caller give a path
    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    # return the model to the caller
    return model


def load_fc_model(ModelClass,
                  pretrained_fc,
                  fc_shape,
                  output_shape,
                  weights_path=None,
                  use_gpu=False):
    """
    :param ModelClass: Class for model to use
    :param pretrained_fc: Model's pretrained fully connected layers
    :param fc_shape: Number of outputs of feature extractor; used as number of inputs for next layer 
    :param output_shape: Number of desired outputs for the network
    :param weights_path: Path to weights file; default: None
    :param use_gpu: Boolean on whether to use GPU; default: False
    """
    # Contruct a new model based on the given inputs
    model = ModelClass(pretrained_fc, fc_shape, output_shape)

    # check if caller wants to use the GPU, place it there
    if use_gpu:
        model.cuda()

    # load the weights for the model's weights if caller give a path
    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    # return the model to the caller
    return model


def save_model(model, weights_path):
    """
    Save the model's state and weights at the given path 

    :param model: A torch model to save
    :param weights_path: Path to save weights file
    """
    torch.save(model.state_dict(), weights_path)


# Predict the results and display the labels with the image
def visualize_model(model, data_set_loader, num_images=5, use_gpu=None):
    """
    Predict the results and display the labels with the image

    :param model: Model to use for predicting results
    :param data_set_loader: Dataset to load the images; any preprocessing need on the data is done here
    :param num_images: Number of images on which to predict results. Use if there are more images in the dataloader than want to predict; default: 5
    :param use_gpu: Boolean on whether to use GPU; default: False
    """
    
    # if caller does not set use_gpu, check and set it here
    if use_gpu is None:
        use_gpu = torch.cuda.is_avalable()

    # load the data, predict attributes, display predictions
    for i, data in enumerate(data_set_loader):
        # extract labels from loaded data
        _, inputs, labels = data

        # Wrap the labels and inputs in Variables so autograd can be done automatically
        if use_gpu:
            # if using GPU, put data on GPU
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        # predict the results
        outputs = model(inputs)
        # get the results from the output
        _, preds = torch.max(outputs.data, 1)

        # plot the results
        # create a new figure
        plt.figure()
        # make sure the input is on the CPU instead of GPU
        imshow(inputs.cpu().data[0])
        # put predictions as title
        plt.title('pred: {}'.format(preds[labels.data[0]]))
        # display the image
        plt.show()
        # break if the number of images to test is reached
        if i == num_images - 1:
            break
