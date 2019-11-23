from scipy import ndimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
    A suite of functions to manipulate numpy image matrices. Manipulations include rotation, additive gaussian noise, speckle,
    up and down sampling, size standardization, to gray scale.
'''
def rotate_image(image, degree):
    return ndimage.rotate(image, degree, reshape=False)

def add_gaussian_noise(image, mean, var):
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_image = image + gauss

    return noisy_image

def rgb2gray(image, channels=1):
    grayscale2D = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    if channels == 1:
        return np.expand_dims(grayscale2D, axis=2)
    if channels == 3:
        return np.repeat(grayscale2D[:, :, np.newaxis], 3, axis=2)    

def add_speckle(image, var):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch) * var #rescales variance of distribution
    gauss = gauss.reshape(row,col,ch)
    noisy_image = image + image * gauss
    return noisy_image

def down_sample_image(image):
    im = tf.image.resize(image, size=[240,340], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=True)
    image_downscaled = tf.dtypes.cast(im, dtype=tf.uint8)
    return image_downscaled

def standardize_image(image, height, width):
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
    image_resized = image.astype(np.float32)

    # if single sample, reshape to include the channel again
    if (len(image_resized.shape) == 2):
        image_resized = image_resized.reshape(*image_resized.shape, 1)
    
    return image_resized

def up_sample_image(image):
    im = tf.image.resize(image, size=[900,1200], method=tf.image.ResizeMethod.BICUBIC)
    image_upscaled = tf.dtypes.cast(im, dtype=tf.uint8)
    return image_upscaled
