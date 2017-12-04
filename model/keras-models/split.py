import os
from shutil import copyfile, rmtree
import numpy as np
from glob import glob
import pandas as pd

def train_valid_split(source_dir, target_dir, labels_file, train_size=0.75, valid_size=0.15):
    """
    Split data into train, validation, and test sets

    :param source_dir: Path to data folder.
    :param target_dir: Path to output folder.
    :param labels_file: Path to file with labels
    :param train_size: Float of training set size
    :param valid_size: Float of validation set size
    """

    # Remove the files if the already exsist
    # Creates a new set of train, valiadation, and test sets each time function is ran
    for folder_name in ['train', 'valid', 'test']:
        rmtree(os.path.join(target_dir, folder_name), ignore_errors=True)
        os.makedirs(os.path.join(target_dir, folder_name))

    # reorder all images in a random order
    filenames = np.random.permutation(glob(os.path.join(source_dir, '*.jpg')))

    # Calculate stop indices for train and validation when splitting dataset
    train_idx = int(len(filenames) * train_size)
    test_idx = int(len(filenames) * (train_size + valid_size))

    # Open the file of landmarks for each image and store in Pandas DataFrame
    landmarks = pd.read_csv(labels_file, header=None, delimiter=' ')
    
    # Create array to hold landmarks locations for each set
    landmarks_train = np.zeros((train_idx, 12))
    landmarks_valid = np.zeros((int(len(filenames)) - train_idx, 12))
    landmarks_test = np.zeros((int(len(filenames)) - test_idx, 12))

    # keep track of how many landmarks are in each set, used to assign landmark to correct image
    counter_train = 0
    counter_valid = 0
    counter_test = 0

    # Iterate thorugh dataset
    for idx, filename in enumerate(filenames):
        # get filename only, remove directory information
        target_name = filename.split('/')[-1]

        # if the current index is less than the stop index, put image into train folder
        if idx < train_idx:
            target_filepath = os.path.join(target_dir, 'train', target_name)
            landmarks_train[counter_train, :] = landmarks.iloc[idx, :]
            counter_train += 1
        # if the current index is between both stop indices, put the image in the validation folder
        elif idx < test_idx and idx >= train_idx:
            target_filepath = os.path.join(target_dir, 'valid', target_name)
            landmarks_valid[counter_valid, :] = landmarks.iloc[idx, :]
            counter_valid += 1
        # All remaining images are part of the test folder
        else:
            target_filepath = os.path.join(target_dir, 'test', target_name)
            landmarks_test[counter_test, :] = landmarks.iloc[idx, :]
            counter_test += 1

        copyfile(filenames[idx], target_filepath)

    # Save landmark location files as csv
    np.savetxt(os.path.join(target_dir, 'train.csv'), landmarks_train.astype('uint64'), fmt='%d', delimiter=' ')
    np.savetxt(os.path.join(target_dir, 'valid.csv'), landmarks_valid.astype('uint64'), fmt='%d', delimiter=' ')
    np.savetxt(os.path.join(target_dir, 'test.csv'), landmarks_test.astype('uint64'), fmt='%d', delimiter=' ')

if __name__ == '__main__':
    train_valid_split('/home/frank/Documents/experimental/data/DeepFashion/output/1', '/home/frank/Desktop/keras-models/data/target', '/home/frank/Documents/experimental/data/DeepFashion/1_landmarks-split.csv')
