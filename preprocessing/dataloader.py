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
        # print('TYRING TO OPEN FILE: {}'.format(os.path.join(self.root_dir, '{}.jpg'.format(str(self.landmarks_frame.ix[idx, 0])))))
        image_name = os.path.join(self.root_dir, '{}.jpg'.format(
            str(self.landmarks_frame.ix[idx, 0])))
        image = Image.open(image_name).convert('RGB')
        landmarks = self.landmarks_frame.ix[idx, 3:14].as_matrix().astype(
            'float32')
        visibility = self.landmarks_frame.ix[idx, 15:].as_matrix().astype(
            'int64')

        if self.transform is not None:
            image = self.transform(image)
            image = image.unsqueeze(0)

        return image, torch.from_numpy(landmarks), torch.from_numpy(visibility)
