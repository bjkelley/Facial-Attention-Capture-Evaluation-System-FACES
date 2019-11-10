import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from scipy import ndimage

def emotion(x):
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

data = []
for i in glob.glob("facesdb/**/bmp/*.bmp", recursive=True):
    data.append(plt.imread(i))

data = np.stack(data)

y = []
image = 0 
for i in range(0,36):
    for j in range(0,10):
        y.append(emotion(j))

dataset = tf.data.Dataset.from_tensor_slices((data, y))