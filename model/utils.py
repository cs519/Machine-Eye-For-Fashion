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
    :rtype pretrained_features: A torch Sequential container
    :rtype pretrained_fc: A torch Linear container
    :rtype pretrained_features: An integer of Linear/Dense layer outputs """

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

    if use_gpu:
        pretrained_features.cuda()

    if pop_last_pool_layer:
        pretrained_features = nn.Sequential(*list(
            pretrained_features.children())[:-1])

    for param in pretrained_features.parameters():
        param.requires_grad = False

    return pretrained_features, pretrained_fc, fc_dim


def load_fcn_model(ModelClass,
                   input_shape,
                   output_shape,
                   weights_path=None,
                   return_conv_layer=False,
                   use_gpu=False):
    model = ModelClass(input_shape, output_shape, return_conv_layer)

    if use_gpu:
        model.cuda()

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    return model


def load_fc_model(ModelClass,
                  pretrained_fc,
                  fc_shape,
                  output_shape,
                  weights_path=None,
                  use_gpu=False):
    model = ModelClass(pretrained_fc, fc_shape, output_shape)

    if use_gpu:
        model.cuda()

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    return model


def save_model(model, weights_path):
    torch.save(model.state_dict(), weights_path)


def visualize_model(model, data_set_loader, num_images=5, use_gpu=None):
    if not use_gpu:
        use_gpu = torch.cuda.is_avalable()

    for i, data in enumerate(data_set_loader):
        batch_files, inputs, labels = data

        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    plt.figure()

    imshow(inputs.cpu().data[0])
    plt.title('pred: {}'.format(dset_classes[labels.data[0]]))
