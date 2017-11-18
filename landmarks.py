import cv2
import os.path as path
import numpy as np


class Landmarks:
    """
    Class to draw landmarks on an image

    :param: image_name: An image array, landmarks will be added to this image
    :param: land_list: List of landmarks associated with the image
    """

    def __init__(self, image_name, land_list):
        self.image_name = image_name
        self.land_list = land_list

    def showLand(self):
        image = cv2.imread(self.image_name)
        n = len(self.land_list)
        for i in range(n):
            x = self.land_list[i][0]
            y = self.land_list[i][1]
            cv2.circle(image, (x, y), 5, (0, 0, 255), thickness=2)
        name = path.split(self.image_name)
        name = path.join("output", name[1])
        cv2.imwrite(name, image)
