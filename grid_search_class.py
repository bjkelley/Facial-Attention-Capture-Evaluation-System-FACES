import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import create_dataset
import functools
import numpy as np
import matplotlib.pyplot as plt
import time
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
            if num in numsUsed: num=(num+1)%5 #guarantees samples not used twice
            if num == 0:
                data.append(image)
                labels.append(label)
            elif num == 1:
                data.append(create_dataset.rgb2gray(image, channels=3))
                labels.append(label)
            elif num == 2:
                newImage = create_dataset.add_speckle(image, 0.05)
                newImage = np.clip(newImage, a_min=0, a_max=1) #ensures normalization is maintained
                data.append(newImage)
                labels.append(label)
            elif num == 3:
                newImage = create_dataset.add_gaussian_noise(image, 0, 0.005)
                newImage = np.clip(newImage, a_min=0, a_max=1) #ensures normalization is maintained
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


# dropout rate, num conv layers, learning rate, normalize data, batch size, regularizer number
def generalizedSVM(input_shape=(128,128, 3), num_conv_layers=3, dropout=0.3, learning_rate=0.001, regularizer=0.01):
    model = keras.Sequential()

    num_filters = 16
    kernel_size = int(2**num_conv_layers)

    model.add(layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                            activation="relu", input_shape=input_shape))
    model.add(layers.Dropout(dropout))

    num_filters = num_filters + 8
    kernel_size = int(kernel_size / 2)

    for layer in range(0, num_conv_layers - 1):
        model.add(layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), strides=4, padding='same', activation="relu"))
        model.add(layers.Dropout(dropout))
        num_filters = num_filters + 8
        kernel_size = int(kernel_size / 2)

    model.add(layers.Flatten())
    model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(regularizer)))
    model.add(layers.Activation('softmax'))

    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = "top3_acc"

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-8)

    model.compile(loss='squared_hinge',
                  optimizer=optimizer,
                  metrics=['accuracy', top3_acc])
    return model


def get_CNN_model(num_conv_layer_groups=3, dropout=0.2, learning_rate=0.001):
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

if __name__ == '__main__':
    dataLoader = create_dataset.DataLoader()
    trainSet = dataLoader.getDataset(num_samples=300, normalize=True)[0] #gives first 30 subjects
    testSet = dataLoader.getDataset(start_index=300, normalize=True)[0] #gives last 6 subjects
    testSet = testSet.batch(10)

    #Apply transformations to expand dataset
    # trainSet = trainSet.repeat(3)
    trainSet = applyTransform(trainSet, num_repeat=5)

    layer_options = [2, 3, 4]
    dropout_options = [0.2, 0.3, 0.4, 0.5]
    lr_options = [0.1, 0.01, 0.001]
    reg_options = [1, 0.1, 0.01]
    batch_options = [15, 30, 60]

    stopCallback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)
    count = 250
    skip = True

    for layer in layer_options:
        for drop in dropout_options:
            for lr in lr_options:
                for reg in reg_options:
                    for batch in batch_options:
                        if skip:
                            if count != 0:
                                count = count - 1
                                continue
                            else:
                                skip = False
                        current_model = generalizedSVM(num_conv_layers=layer, dropout=drop, learning_rate=lr, regularizer=reg)
                        current_trainSet = trainSet.shuffle(200).batch(batch)
                        history = current_model.fit(current_trainSet, epochs=100, validation_data=testSet, callbacks=[stopCallback])

                        fig = plt.figure(figsize=(10.8, 7.2), dpi=100)
                        plt.plot(history.history['accuracy'], label="train acc")
                        plt.plot(history.history['val_accuracy'], label="test acc")
                        plt.plot(history.history['top3_acc'], label="train top3")
                        plt.plot(history.history['val_top3_acc'], label="test top3")
                        plt.title(f"{history.history['val_accuracy'][-1]:.4f}% accuracy generalizedSVM: layers={layer}, dropout={drop}, lr={lr}, reg={reg}, batch={batch}")
                        plt.xlabel("epochs")
                        plt.ylabel("accuracy")
                        plt.legend()
                        plt.savefig(f"./figures/{history.history['val_accuracy'][-1]:.4f}GeneralizedSVM#l{layer}-d{drop}-lr{lr}-r{reg}-b{batch}.png")
                        plt.close()

    # for layer in layer_options:
    #     for drop in dropout_options:
    #         for lr in lr_options:
    #             for batch in batch_options:
    #                 current_model = get_CNN_model(num_conv_layer_groups=layer, dropout=drop, learning_rate=lr)
    #                 current_trainSet = trainSet.shuffle(200).batch(batch)
    #                 history = current_model.fit(current_trainSet, epochs=65, validation_data=testSet)
    #
    #                 fig = plt.figure(figsize=(10.8, 7.2), dpi=100)
    #                 plt.plot(history.history['accuracy'], label="train acc")
    #                 plt.plot(history.history['val_accuracy'], label="test acc")
    #                 plt.plot(history.history['top3_acc'], label="train top3")
    #                 plt.plot(history.history['val_top3_acc'], label="test top3")
    #                 plt.title(f"{history.history['val_accuracy'][-1]:.4f}% accuracy CNN2: layers={layer}, dropout={drop}, lr={lr}, reg={reg}, batch={batch}")
    #                 plt.xlabel("epochs")
    #                 plt.ylabel("accuracy")
    #                 plt.legend()
    #                 plt.savefig(f"./figures/{history.history['val_accuracy'][-1]:.4f}CNN2#l{layer}-d{drop}-lr{lr}-r{reg}-b{batch}.png")
