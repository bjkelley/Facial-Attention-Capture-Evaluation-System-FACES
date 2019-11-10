import glob, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy import ndimage

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

    def getDataset(self):
        datapath = os.path.join(self.datapath, "facesdb/**/bmp/*.bmp")
        data = []
        for i in glob.glob(datapath, recursive=True):
            data.append(plt.imread(i))

        data = np.stack(data)
        y = self.create_label_vector()
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

dataLoader = DataLoader()
dataset = dataLoader.getDataset()

'''Debugging purpose
for img, label in dataset:
    plt.imshow(img.numpy())
    plt.show()
'''
