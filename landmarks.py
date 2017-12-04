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
        image = cv2.imread(self.image_name) #read image
        n = len(self.land_list) #get amount of landmarks
        #loop through list writing landmarks
        for i in range(n):
            x = self.land_list[i][0] #x coordinate
            y = self.land_list[i][1] #y coordinate
            cv2.circle(image, (x, y), 5, (0, 0, 255), thickness=2) #create circle on landmark
        name = path.split(self.image_name) #removing path from image
        name = path.join("output", name[1]) #adding output path to image
        cv2.imwrite(name, image) #write image
