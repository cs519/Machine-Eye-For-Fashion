import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class FashionLandmarkDataset(Dataset):
    """Fashion Landmark dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        """
        Load Landmark data loader
        :param csv_file (string): Path to the csv file with annotations
        :param root_dir (string): Directory with all the images
        :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.landmarks_frame = pd.read_csv(csv_file, header=None, delimiter=' ')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Return the len of the DataFrame
        """
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        """
        Return the item at the current index
        
        function is not called explicitly
        """

        # get the next image to open from the landmarks dataframe
        image_name = os.path.join(self.root_dir, '{}.jpg'.format(
            str(self.landmarks_frame.ix[idx, 0])))
        # open the image with Pillow
        image = Image.open(image_name).convert('RGB')
        # read the landmark as a float
        landmarks = self.landmarks_frame.ix[idx, 3:14].as_matrix().astype(
            'float32')
        # read the visiability of the landmarks as integers
        visibility = self.landmarks_frame.ix[idx, 15:].as_matrix().astype(
            'int64')
        
        # if client passed a transform when constructing opbject, transform the image
        if self.transform is not None:
            image = self.transform(image)
            # unsqueeze to match the expected input of first layer of network
            image = image.unsqueeze(0)

        # return the image (transform change image to tensor), landmarks in a tensor, and visibility in tensor 
        return image, torch.from_numpy(landmarks), torch.from_numpy(visibility)
