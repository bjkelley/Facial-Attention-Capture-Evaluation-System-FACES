from create_dataset import *

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras import backend as K
import functools as functools
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# pixel height/width for resizing images
HEIGHT = WIDTH = 60

emotion_to_onehot = {
    'neutral frontal': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'joy':             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'sadness':         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'surprise':        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'anger':           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'disgust':         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'fear':            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'opened':          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'closed':          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'kiss':            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

# reverse dictionary to be able to get predicted label
onehot_to_emotion = {str(onehot_value): emotion for emotion, onehot_value in emotion_to_onehot.items()}

def get_pred_emotion(one_hot_output):
    '''
    takes in one hot predicted label, outputs the emotion string
    '''
    return onehot_to_emotion[str(one_hot_output)]


def apply_transformations(img):
    '''
    input: img matrix of a single image
    
    ~ applies an image reshape (to 240x340), random rotation, gaussian noise, grayscale
    
    output: transformed image
    '''
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces, extract them
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    x, y, w, h = faces[0]
    img = gray[y:y+h, x:x+w]
    
    # reshape so that there's 1 channel
    img = img.reshape(h, w, 1)
    
    # add gaussian noise, random rotation, and scale to 100x100
    img = add_gaussian_noise(img, 0, 0.2)
    img = rotate_image( img, np.random.randint(-10, 10) )
    img = standardize_image(img, HEIGHT, WIDTH)
    
    return img

def get_train_test_data(num_train_faces=28, oversample_amt=4):
    '''
    uses DataLoader API to load train/test data

    input: 
        - num_train_faces, up to 36
           (our data has 10 emotions for each person, and there are 36 unique
            people. this function splits 28 (by default) unique people into
            training, and puts the rest into testing)

        - oversample_amt
            (number of times to augment each image in the training data,
             to increase the amount of data we have for training. this
             will not affect the testing data, or use people in that dataset.)

    output: 
        - train_X, train_Y
        - test_X, test_Y
    '''
    # calculate index to split train/test data
    split_idx = num_train_faces * 10

    # read in data
    dataLoader = DataLoader()
    dataset, data, labels = dataLoader.getDataset()

    # separate train and test
    train_X = data[:split_idx,:,:,:]
    train_Y = labels[:split_idx]

    test_X = data[split_idx:,:,:,:]
    test_Y = labels[split_idx:]

    # oversample the training dataset 
    for i in range(oversample_amt):
        train_X = np.append(train_X, data[:split_idx,:,:,:], axis=0)
        train_Y = np.append(train_Y, labels[:split_idx], axis=0)

    return train_X, train_Y, test_X, test_Y


def transform_data(X, Y):
    '''
    takes in X and Y, applies transformations and one-hot-encoding
    '''
    ## format X array (apply image augmentation and resize)
    X_transformed = np.array([ np.array(apply_transformations(img)).reshape(HEIGHT,WIDTH,1) 
                                for img in X ])

    # format lables to be one-hot-encoded
    onehot_lables = [emotion_to_onehot[emotion] for emotion in Y]
    one_hot_Y = np.array(onehot_lables)

    return X_transformed, one_hot_Y

def fastSVM(input_shape=(HEIGHT,WIDTH, 1)):
	model = keras.Sequential()
	model.add(layers.Conv2D(filters=16, kernel_size=(4,4), strides=1,
		padding='same', activation="relu", input_shape=input_shape))
	model.add(layers.Conv2D(filters=16, kernel_size=(4,4), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=24, kernel_size=(2,2), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=32, kernel_size=(2,2), strides=2, padding='same', activation="relu"))	
	model.add(layers.Flatten())
	model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.01)))
	model.add(layers.Activation('softmax'))

	top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
	top3_acc.__name__ = "top3_acc"
	model.compile(loss='squared_hinge',
	              optimizer='adam',
	              metrics=['accuracy', top3_acc])
	return model

def get_CNN_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                                input_shape=(HEIGHT, WIDTH, 1)))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
   
   # define custom top3 accuracy metric
    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'
   
    # compile model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', top3_acc])

    return model

def generate_and_save_figures(history):
    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Train/test accuracy over training')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('figures/train_vs_test_accuracy.png')
    plt.cla()

    # top3 accuracy
    plt.plot(history.history['top3_acc'])
    plt.plot(history.history['val_top3_acc'])
    plt.title('Train/test top3 accuracy over training')
    plt.ylabel('top3 accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('figures/train_vs_test_top3_accuracy.png')
    plt.cla()

    # loss 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train/test loss over training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('figures/train_vs_test_loss.png')
    plt.cla()



def main():

    # get train/test data
    train_X, train_Y, test_X, test_Y = get_train_test_data(oversample_amt=4)

    # trainsform sets
    train_X, train_Y = transform_data(train_X, train_Y)
    test_X, test_Y = transform_data(test_X, test_Y)

    # print shapes
    print("shape of train_X: ", train_X.shape)
    print("shape of train_Y: ", train_Y.shape)

    print("shape of test_X: ", test_X.shape)
    print("shape of test_Y: ", test_Y.shape)

    # get model
    model = get_CNN_model()
    model = fastSVM()

    # run on dataset
    history = model.fit(train_X, train_Y, 
                        epochs=25, batch_size=200, 
                        validation_data=(test_X,test_Y))

    # save figures into figures/ directory
    #generate_and_save_figures(history)

    #model.save('trained_models/cnn2.h5')


main()
