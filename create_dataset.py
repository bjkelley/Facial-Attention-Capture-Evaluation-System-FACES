import glob, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy import ndimage
from image_mod_functions import rotate_image, add_gaussian_noise, add_speckle, rgb2gray

class DataLoader:
    def __init__(self, datapath = ""):
        self.datapath = datapath

    def get_emotion(self, x):
        return {
            0: [1,0,0,0,0,0,0,0,0,0],#'neutral frontal',
            1: [0,1,0,0,0,0,0,0,0,0],#'joy',
            2: [0,0,1,0,0,0,0,0,0,0],#'sadness',
            3: [0,0,0,1,0,0,0,0,0,0],#'surprise',
            4: [0,0,0,0,1,0,0,0,0,0],#'anger',
            5: [0,0,0,0,0,1,0,0,0,0],#'disgust',
            6: [0,0,0,0,0,0,1,0,0,0],#'fear',
            7: [0,0,0,0,0,0,0,1,0,0],#'opened',
            8: [0,0,0,0,0,0,0,0,1,0],#'closed',
            9: [0,0,0,0,0,0,0,0,0,1],#'kiss'
        }[x]

    def getDataset(self):
        # each entries of datset is a tuple of (image, label)
        datapath = os.path.join(self.datapath, "facesdb/**/bmp/*.bmp")
        data = []
        for i in sorted(glob.glob(datapath, recursive=True)):
            data.append(plt.imread(i))

        data = np.stack(data)
        y = self.create_label_vector()
        print(type(data))
        data = tf.cast(data, tf.float16)
        dataset = tf.data.Dataset.from_tensor_slices((data, y))
        return dataset

    def create_label_vector(self):
        # y is label vector
        y = []
        image = 0
        for i in range(0,36):
            for j in range(0,10):
                y.append(self.get_emotion(j))

        return y

'''Testing'''
'''
dataLoader = DataLoader()
dataset = dataLoader.getDataset()

for img, label in dataset:
    img = rgb2gray(img)
    img = add_speckle(img)
    img = add_gaussian_noise(img, 0, 0.2)
    img = rotate_image(img, 45)
    plt.imshow(np.squeeze(img))
    plt.show()
'''
