import os
import cv2
import pandas as pd

from scipy.misc import imsave, imread
from keras.models import load_model
from skimage.transform import resize

import matplotlib.pyplot as plt
import argparse

from train import create_model

def draw_landmarks(img, landmarks):
    """
    Draw landmarks on image
    :param img: image to draw on
    :param landmarks: array of landmarks
    :rtype img: image with landmarks drawn
    """

    # iterate through landmarks
    for i in range(0, landmarks.shape[1], 2):
        # get the x and y values of each landmark
        x = landmarks[0][i]
        y = landmarks[0][i+1]
        # draw a circle at the location of the landmark
        cv2.circle(img, (x,y), 5, (0,0,255), thickness=2)

    # return an image with landmarks drawn
    return img

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='results/weights_9-3691.9892.hdf5')
    parser.add_argument('--image', type=str, default='data/test-1/10.jpg')
    args = parser.parse_args()

    # create the model
    model = create_model(12)

    # Check if weights exists, if not exit
    if not os.path.exists(args.weights):
        print('Weights file does not exist')
        exit(1)
    # load the model's weights
    model.load_weights(args.weights)

    # read the test image
    image = imread(args.image)
    # get the height and width of the test image
    r, c, _ = image.shape

    
    # resize the image to (224, 224, 3) so it matches size of training data to try to get a better result
    image = resize(image, output_shape=(224,224,3))
    
    # predit the landmark locations
    # reshape the image to make expected input shape of model
    prediction = model.predict(image.reshape(-1, 224,224,3))

    # remap the predicted location to image's orignal size (optional)
    # prediction[::2] = prediction[::2] * c // 224
    # prediction[1::2] = prediction[1::2] * r // 224

    # print prediction
    print(prediction)

    # display the image with drawn landmarks
    plt.imshow(draw_landmarks(image, prediction))
    plt.show()

    # save the image with drawn landmarks
    imsave('test.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    test()
