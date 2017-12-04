import numpy as np
from scipy.misc import imread
import torch

from model.utils import get_pretrained_model
from model.models import AttributeFCN, predict_attributes, create_attributes_fcn_model
from preprocessing.preprocessing import image_loader, load_label_values, get_attribute_dims

# set path of test image
image_paths = ['demo_data/test/test_nan_01.png']
# check if an NVIDIA GPU is available, if so, use it
use_gpu = torch.cuda.is_available()
# set train and validation directories
TRAIN_IMAGES_FOLDER = 'data/ClothingAttributeDataset/train/'
VALID_IMAGES_FOLDER = 'data/ClothingAttributeDataset/valid/'
# set the path for the labels for each image, and the path to the labels to predict on
labels_file = 'data/labels.csv'
label_values_file = 'data/label_values.json'

# get the base model up to block5
pretrained_conv_model, _, _ = get_pretrained_model(
    'vgg16', pop_last_pool_layer=True, use_gpu=use_gpu)

# load the model to use for prediction
target_dims = get_attribute_dims(label_values_file)
attribute_models = create_attributes_fcn_model(
    AttributeFCN,
    512,
    pretrained_conv_model,
    target_dims,
    'weights/fcn/',
    labels_file,
    TRAIN_IMAGES_FOLDER,
    VALID_IMAGES_FOLDER,
    num_epochs=1,
    is_train=False,
    use_gpu=use_gpu)

# predict the attribues for each image
for image_path in image_paths:
    results = predict_attributes(
        image_path,
        pretrained_conv_model,
        attribute_models,
        flatten_pretrained_out=False)

    print(results)


