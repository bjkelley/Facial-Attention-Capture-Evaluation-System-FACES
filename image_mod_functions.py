from scipy import ndimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, degree):
    return ndimage.rotate(image, degree, reshape=False)

def add_gaussian_noise(image, mean, var):
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_image = image + gauss

    return noisy_image.numpy()

def rgb2gray(image, channels=1):
    grayscale2D = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    if channels == 1:
        return grayscale2D
    if channels == 3:
        return np.repeat(grayscale2D[:, :, np.newaxis], 3, axis=2)
    

def add_speckle(image, var):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch) * var #rescales variance of distribution
    gauss = gauss.reshape(row,col,ch)
    noisy_image = image + image * gauss
    return noisy_image

def standardize_image(image, height, width):
    im = tf.image.resize(image, size=[height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
    image_downscaled = tf.dtypes.cast(im, dtype=tf.uint8)
    return image_downscaled