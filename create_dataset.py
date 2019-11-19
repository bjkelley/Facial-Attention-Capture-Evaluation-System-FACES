import glob, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy import ndimage
import cv2
from image_mod_functions import rotate_image, add_gaussian_noise, add_speckle, rgb2gray, standardize_image

class DataLoader:
    def __init__(self, datapath = ""):
        self.datapath = datapath

    def get_emotion(self, x):
        return {
            0: 'neutral frontal',
            1: 'joy',
            2: 'sadness',
            3: 'surprise',
            4: 'anger',
            5: 'disgust',
            6: 'fear',
            7: 'opened',
            8: 'closed',
            9: 'kiss'
        }[x]

    def get_one_hot(self, x):
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

    def getDataset(self, start_index=0, num_samples=0):
        # each entries of datset is a tuple of (image, label)
        datapath = os.path.join(self.datapath, "facesdb/**/bmp/*.bmp")
        data = []
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        for i in sorted(glob.glob(datapath, recursive=True)):
            image = plt.imread(i)
            # print(type(image))
            # print(image.dtype)
            gray = cv2.cvtColor(image[0:-40,:,:], cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1,4)
            x, y, w, h = faces[0]
            image = image[y:y+h,x:x+w,:]
            # plt.imshow(image)
            # plt.title(i)
            # plt.show()
            image = tf.cast(standardize_image(image, 128, 128), tf.float32) / 255.0 #normalize to float on [0, 1]
            data.append(image)

        data = np.stack(data)
        y = self.create_label_vector()
        # data = tf.cast(data, tf.float32)
        if num_samples != 0:
            end_index = start_index + num_samples
            dataset = tf.data.Dataset.from_tensor_slices((data[start_index:end_index], y[start_index:end_index]))
        elif start_index !=0:
            dataset = tf.data.Dataset.from_tensor_slices((data[start_index:], y[start_index:]))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((data, y))
        return dataset

    def create_label_vector(self):
        # y is label vector
        y = []
        image = 0
        for i in range(0,36):
            for j in range(0,10):
                y.append(self.get_one_hot(j))

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
