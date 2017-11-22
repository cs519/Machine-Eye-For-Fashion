import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io

class FashionLandmarkDataset(Dataset):
    """Fashion Landmark dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """

        self.landmarks_frame = pd.read_csv(csv_file, header=None, delimiter=' ')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.landmarks_frame, '.jpg')
        image = io.imread(image_name)
        landmarks = self.landmarks_frame.ix[idx, 2:13].as_matrix().astype('uint16')
        visibility = self.landmarks_frame.ix[idx, 14:].as_matrix().astype('uint8')
        sample = {'image': image, 'landmarks':landmarks, 'visibiliy': visibility}

        if self.transform:
            sample = self.transform(sample)

        return sample