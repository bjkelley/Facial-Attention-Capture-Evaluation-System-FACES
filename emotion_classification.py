from create_dataset import *

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import cv2 as cv2

# map emotion to onehot value
# can do this in code if we want to, but just wrote this to be explicit

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
    img = standardize_image(img)
    
    return img

## read in data
dataLoader = DataLoader()
dataset, data, labels = dataLoader.getDataset()

## format X array
X_transformed = np.array([ np.array(apply_transformations(img)).reshape(240,340,1) 
                          for img in data])

## format lables to be one-hot-encoded
onehot_lables = [emotion_to_onehot[emotion] for emotion in labels]
one_hot_Y = np.array(onehot_lables)

print("shape of X_transformed: ", X_transformed.shape)
print("shape of one_hot_Y: ", one_hot_Y.shape)

#create model
model = Sequential([
    Conv2D(64, kernel_size=(8,8), activation='relu', input_shape=(240,340,1), 
           kernel_regularizer=keras.regularizers.l2(0.01)),
    #Conv2D(64, kernel_size=(8, 8), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2), strides=(2, 2)),
    Dropout(0.2),
    
    Conv2D(128, kernel_size=(8, 8), activation='relu'),
    #Conv2D(128, kernel_size=(8, 8), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_transformed, one_hot_Y, 
          epochs=10, batch_size=200, validation_split=0.20)

