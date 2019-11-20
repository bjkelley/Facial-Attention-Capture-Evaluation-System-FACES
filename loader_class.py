import tensorflow.keras as keras
import cv2 as cv2
import numpy as np
import functools as functools

from image_mod_functions import standardize_image

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

class LoadModel():
    '''
    LoadModel: class to load pretrained image emotion classifier
    contains functions to:
        - return predicted emotion string
        - preprocess an image to get it ready for classifiaction
        - return top prediction and/or top3 predictions
    '''
    def __init__(self, model_type="cnn2", input_shape=(60,60, 1)):
        # initialize object attributes
        self.model_type = model_type.lower()
        self.input_shape = input_shape

        # load OpenCV face detection cascade for preprocessing steps
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # load in correct model
        if self.model_type == "cnn2":
            self.model = keras.models.load_model('trained_models/cnn2.h5', custom_objects=dependencies)
        elif self.model_type == "cnn1":
            self.model = keras.models.load_model('trained_models/cnn2.h5', custom_objects=dependencies)
        elif self.model_type == "fastsvm":
            self.model = keras.models.load_model('trained_models/fastsvm.h5', custom_objects=dependencies)
        else:
            raise Exception(f"Model type {model_type} not found. Try another model.")

    def get_emotion(self, one_hot_idx):
        '''returns emotion string that corresponds to the one-hot-encoded index'''
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
        }[one_hot_idx]

    def preprocess(self, img):
        '''preprocesses image accordingly so that model can take it as input
        - use OpenCV face detector and convert to grayscale
        - resize image
        '''

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces, extract them
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        x, y, w, h = faces[0]
     
        # reshape so that there's self.input_shape[2] channels if necessary
        if (self.input_shape[2] != 1):
            img = img[y:y+h, x:x+h, :]
        
        # else, use grayscale
        else:
            img = gray[y:y+h, x:x+w].reshape(h, w, 1)

        # scale image to input_shape width x height
        img = standardize_image(img, self.input_shape[0], self.input_shape[1])

        return np.array([img])
    
    def classify(self, img, k_most_confident_classes=3):
        '''takes in img
        returns: 
            - predicted class string
            - top k (3 by default) most confident emotion string predictions
            - the corresponding k (3 by default) emotion probailities
        '''

        # plug into model to get array of probabilities
        probs = self.model.predict_proba(img).reshape(-1)
        
        # sort probability indices from highest to lowest probabilitiy
        sorted_indices = np.argsort(probs)[::-1]

        # get sorted emotions
        sorted_emotions = [self.get_emotion(idx) for idx in sorted_indices]

        # get sorted probabilities
        sorted_probabilities = [probs[idx] for idx in sorted_indices]

        return (sorted_emotions[0], 
                sorted_emotions[:k_most_confident_classes],
                sorted_probabilities[:k_most_confident_classes]
               )