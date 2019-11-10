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

    return noisy_image.numpy()

def rgb2gray(image):
    grayscale2D = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    return np.expand_dims(grayscale2D, axis=2)

def add_speckle(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy_image = image + image * gauss
    return noisy_image

def add_saltnpepper_noise(image):
    out = np.zeros(image.shape)
    # salt coordinates
    coords = [np.random.randint(0,26,50), np.random.randint(0,26,50)]

    # mask - 0 are regions where salt can be applied, otherwise don't touch
    mask = np.zeros(out.shape)
    mask[:13,:13] = 1
    mask[-13:,-13:] = 2

    # where does the salt coordinates land on the mask
    a = mask[coords]

    # find points where mask is 0
    b = np.nonzero(a==0)

    # copy from coords only where mask is 0
    valid_coords = np.array(coords)[:,b]

    # apply salt on valid coordinates
    out = out.numpy()
    out[valid_coords.tolist()]=1
