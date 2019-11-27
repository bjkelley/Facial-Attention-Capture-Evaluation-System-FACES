import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import create_dataset
import functools
import numpy as np
from tensorflow.keras import models

# pixel height/width for resizing images
HEIGHT = WIDTH = 60


def applyTransform(dataset, num_repeat=3):
    data = []
    labels = []
    for image, label in dataset:
        randNums = np.random.randint(low=1, high=5, size=num_repeat)
        numsUsed = []
        for num in randNums:
            if num in numsUsed: num = (num + 1) % 5  # guarantees samples not used twice
            if num == 0:
                data.append(image)
                labels.append(label)
            elif num == 1:
                data.append(create_dataset.rgb2gray(image, channels=3))
                labels.append(label)
            elif num == 2:
                newImage = create_dataset.add_speckle(image, 0.05)
                newImage = np.clip(newImage, a_min=0, a_max=1)  # ensures normalization is maintained
                data.append(newImage)
                labels.append(label)
            elif num == 3:
                newImage = create_dataset.add_gaussian_noise(image, 0, 0.005)
                newImage = np.clip(newImage, a_min=0, a_max=1)  # ensures normalization is maintained
                data.append(newImage)
                labels.append(label)
            elif num == 4:
                angle = np.random.randint(low=1, high=7, size=1)[0]
                newImage = create_dataset.rotate_image(image, angle)
                data.append(newImage)
                labels.append(label)
            elif num == 5:
                data.append(image)
                labels.append(label)
            else:
                print("Weird error")
            numsUsed.append(num)
    return tf.data.Dataset.from_tensor_slices((data, labels))


# define an SVM with the given parameters
# dropout rate, num conv layers, learning rate, normalize data, batch size, regularizer number
def get_generalizedSVM(input_shape=(128, 128, 3), num_conv_layers=3, dropout=0.3, learning_rate=0.001, regularizer=0.01):
    model = keras.Sequential()

    num_filters = 16
    kernel_size = int(2 ** num_conv_layers)

    model.add(layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                            activation="relu", input_shape=input_shape))
    model.add(layers.Dropout(dropout))

    num_filters = num_filters + 8
    kernel_size = int(kernel_size / 2)

    for layer in range(0, num_conv_layers - 1):
        model.add(layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), strides=4, padding='same',
                                activation="relu"))
        model.add(layers.Dropout(dropout))
        num_filters = num_filters + 8
        kernel_size = int(kernel_size / 2)

    model.add(layers.Flatten())
    model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(regularizer)))
    model.add(layers.Activation('softmax'))

    # keep track of the top 3 accuracy
    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = "top3_acc"

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    model.compile(loss='squared_hinge',
                  optimizer=optimizer,
                  metrics=['accuracy', top3_acc])
    return model


# define a CNN model with the given parameters
def get_CNN(num_conv_layer_groups=3, dropout=0.2, learning_rate=0.001):
    model = models.Sequential()

    num = 32

    model.add(layers.Conv2D(num, (3, 3), padding='same', activation='relu',
                            input_shape=(HEIGHT, WIDTH, 1)))
    model.add(layers.Conv2D(num, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(num, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout))

    num = num * 2

    for layer in range(0, num_conv_layer_groups - 1):
        model.add(layers.Conv2D(num, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(num, (3, 3), padding='same', activation='relu'))
        model.add(layers.Conv2D(num, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(dropout))

        num = num * 2

    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # define custom top3 accuracy metric
    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    # compile model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top3_acc])

    return model


def reduceSize(dataset):
    data = []
    labels = []
    for image, label in dataset:
        image = create_dataset.rgb2gray(image)
        data.append(create_dataset.standardize_image(image, HEIGHT, WIDTH))
        labels.append(label)
    return tf.data.Dataset.from_tensor_slices((data, labels))
