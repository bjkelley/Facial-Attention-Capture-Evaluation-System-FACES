from scipy import ndimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, degree):
    return ndimage.rotate(image, degree)

def add_gaussian_noise(image, mean, var):
    sigma = var**0.7
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_image = image + gauss

    return noisy_image

def rgb2gray(image):
    grayscale2D = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    return np.expand_dims(grayscale2D, axis=2)

def add_speckle(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy_image = image + image * gauss
    return noisy_image

def down_sample_image(image):
    im = tf.image.resize(image, size=[240,340], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
    image_downscaled = tf.dtypes.cast(im, dtype=tf.uint8)
    return image_downscaled

def standardize_image(image):
    im = tf.image.resize(image, size=[200,200], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
    image_downscaled = tf.dtypes.cast(im, dtype=tf.uint8)
    return image_downscaled

def up_sample_image(image):
    im = tf.image.resize(image, size=[900,1200], method=tf.image.ResizeMethod.BICUBIC)
    image_upscaled = tf.dtypes.cast(im, dtype=tf.uint8)
    return image_upscaled
