from scipy import ndimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, degree):
    return ndimage.rotate(image, degree)

def add_gaussian_noise(image, mean, var):
    # mean = 0
    # var = 0.2
    sigma = var**0.7
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_image = image + gauss

    return noisy_image.numpy()

def add_salt_n_pepper(image):
    #row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * 1000 * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* 1500 * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0
    
    return out
