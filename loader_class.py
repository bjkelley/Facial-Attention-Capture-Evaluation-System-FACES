from create_dataset import *

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
import functools as functools
import warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from image_mod_functions import standardize_image

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

new_model = 

class LoadModel():
	"""doc"""
	def __init__(self, model_type="cnn2", input_shape=(60,60, 1)):
		model_type = model_type.lower()
		if model_type == "generalizedsvm":
			self.model = tf.keras.models.load_model('generalizedSVM.h5')
		elif model_type == "fastsvm":
			self.model = tf.keras.models.load_model('fastSVM.h5')
		elif model_type == "cnn2"
			self.model = keras.models.load_model('trained_models/cnn2.h5', custom_objects=dependencies)
		else:
			raise Exception(f"Model type {model_type} not found. Try another model.")

	def max_list(result, num_classes=3):
		maxList = []
		for index, probability in enumerate(result):
			if len(MaxList) == 0: # first iteration
				maxList.append((probability, index))
				continue
			#iterate through sorted subset of result
			for rank, element in enumerate(maxList):
				if probability > element:
					maxList.insert((probability, index), rank)
			if len(maxList) > num_classes: #keeps only top num_classes in result
				maxList.pop()
		return maxList

	def get_emotion(x):
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
		}[x]

	def classify(input):
		classes = []
		if len(input.shape) == 4: #handle batch prediction
			result = model.predict(input)
			for prediction in result:
				classes.append(TopThree(prediction))
		elif len(input_shape) == 3: #handle single prediction
			result = model.predict((input)) 
			classes.append(TopThree(result))
		return classes
