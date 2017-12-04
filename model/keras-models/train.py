import os
import numpy as np
import pandas

from scipy.misc import imread
from skimage.transform import resize
from glob import glob

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, GaussianNoise, Dropout, Activation, MaxPooling2D, BatchNormalization, Flatten, Input
from keras.optimizers import Adam
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


def euclidean_loss(y_true, y_pred):
    # Define euclidean loss (1/2N) * Sum ((predictions - targets)^2)
    return K.mean(K.square(y_pred - y_true), axis=-1) / 2


def create_model(num_outputs):
    """ Create regressor to find landmark locations

    :param num_outputs: number of landmark locations
    """
    # Base model is pretrained VGG16 trained on ImageNet
    model_base = VGG16(weights='imagenet', input_shape=(224,224,3))

    # Set the pretrained layers' weights to not be updated
    for layer in model_base.layers:
        layer.trainable = False

    # Create remaining layers:
    # Only take the first 4 blocks of VGG
    x = model_base.get_layer('block4_pool').output
    # Create 5th block of convolutions
    # architecture is similar to VGG16's block5_pool
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    # Flatten everything
    x = Flatten()(x)
    # Create the fully connected layers with relu activation
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    # Output layer is fully connected, and output equal to number of landmarks
    locations = Dense(num_outputs)(x)

    # create the model 
    model_final = Model(inputs=model_base.input, outputs=locations)
    # compile the model with Adam optimizer, and euclidean loss
    model_final.compile(Adam(lr=0.01), loss=euclidean_loss)
    # print model summary
    print(model_final.summary())
    
    # return the model to caller
    return model_final

def load_data(source_dir, mode='train'):
    """
    Load training and validation data

    :param source_dir: path to root data directory
    """
    # open landmarks folders
    train_csv = pandas.read_csv(os.path.join(source_dir, 'train.csv'), header=None, delimiter=' ')
    valid_csv = pandas.read_csv(os.path.join(source_dir, 'valid.csv'), header=None, delimiter=' ')

    # create arrays to hold training and validation images
    imgs_train = np.zeros((train_csv.shape[0], 224, 224, 3))
    imgs_valid = np.zeros((valid_csv.shape[0], 224, 224, 3))

    # get list of training and validating images
    filenames_train = glob(os.path.join(source_dir, '*.jpg'))
    filenames_valid = glob(os.path.join(source_dir, '*.jpg'))

    
    for i, filename in enumerate(filenames_train):
        # load image
        img = imread(os.path.join(source_dir, 'train', filename))
        # resize to (224, 224, 3) to match imput_shape of VGG16
        imgs_train[i,:,:,:] = resize(img, output_shape=(224,224,3), mode='constant')
        # scale landmark location to match the resized image
        train_csv.iloc[i, ::2] = train_csv.iloc[i, ::2] * 224 // img.shape[1]
        train_csv.iloc[i, 1::2] = train_csv.iloc[i, 1::2] * 224 // img.shape[0]

    for i, filename in enumerate(filenames_valid):
        # load image
        img = imread(os.path.join(source_dir, 'valid', filename))
        # resize to (224, 224, 3) to match imput shape of VGG16
        imgs_valid[i,:,:,:] = resize(img, output_shape=(224,224,3), mode='constant')
        # scale landmakr location to match the resized image
        valid_csv.iloc[i, ::2] = valid_csv.iloc[i, ::2] * 224 // img.shape[1]
        valid_csv.iloc[i, 1::2] = valid_csv.iloc[i, 1::2] * 224 // img.shape[0]

    # return loaded data to caller
    return imgs_train, train_csv.values, imgs_valid, valid_csv.values
    

if __name__ == '__main__':
    # seed numpy's random to get similar results after each run
    seed = 7
    np.random.seed(seed=seed)
    # load training and validation data
    data_train, targets_train, data_valid, targets_valid = load_data('data')

    # create the model
    model = create_model(targets_train.shape[1])
    # save the model architecture  (weights, configs, and state)
    model.save('model.h5')

    # create string for weights' name
    checkpoint_weights_filename = 'results/weights_{epoch}-{val_loss:.4f}.hdf5'

    # add callbacks, used when training
    callbacks = [
        # visualation with TensorBoard
        TensorBoard(),
        # check models validation accuracy after 10 epochs
        ModelCheckpoint(checkpoint_weights_filename, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=10),
        # stop the training if model does not imporve after 20 epochs
        EarlyStopping(monitor='val_loss', patience=20)
        ]
    # fit the model
    # pass the training and validation data and targets, and callbacks
    model.fit(x=data_train, y=targets_train, batch_size=32, epochs=1000, validation_data=(data_valid, targets_valid), callbacks=callbacks)

