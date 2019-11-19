import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from image_mod_functions import standardize_image

def fastSVM(input_shape=(128,128, 3)):
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

def generalizedSVM(input_shape=(128,128, 3)):
	model = keras.Sequential()
	model.add(layers.Conv2D(filters=16, kernel_size=(8,8), strides=1,
		padding='same', activation="relu", input_shape=input_shape))
	model.add(layers.Dropout(0.3))
	model.add(layers.Conv2D(filters=24, kernel_size=(4,4), strides=4, padding='same', activation="relu"))	
	model.add(layers.Dropout(0.3))
	model.add(layers.Conv2D(filters=32, kernel_size=(2,2), strides=4, padding='same', activation="relu"))
	model.add(layers.Dropout(0.3))
	model.add(layers.Flatten())
	model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.01)))
	model.add(layers.Activation('softmax'))

	top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
	top3_acc.__name__ = "top3_acc"
	model.compile(loss='squared_hinge',
	              optimizer='adam',
	              metrics=['accuracy', top3_acc])
	return model


class ReadyModel():
	"""docstring for ClassName"""
	def __init__(self, model_type="cnn", input_shape=(128,128, 3)):
		model_type = model_type.lower()
		if model_type == "generalizedsvm":
			self.model = tf.keras.models.load_model('generalizedSVM.h5')
		elif model_type == "fastsvm":
			self.model = tf.keras.models.load_model('fastSVM.h5')
		else:
			raise Exception(f"Model type {model_type} not found. Try another model.")

	def MaxList(result, num_classes=3):
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

	def GetEmotion(x):
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
