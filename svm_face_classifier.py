import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import create_dataset
import image_mod_functions #just for testing

def CreateSVM(units=3, num_filters=15, conv_layers=1):
	model = keras.Sequential()
	model.add(layers.Conv2D(filters=num_filters, kernel_size=(4,4), strides=2,
		padding='same', activation="relu", input_shape=(480,640, 3)))
	for index in range(1, units):
		for sub_layer in range(0,conv_layers-1):
			model.add(layers.Conv2D(filters=num_filters, kernel_size=(2*index+4,2*index+4), strides=1, padding='same', activation="relu"))
		# Downsampling layer
		num_filters = num_filters - 4
		model.add(layers.Conv2D(filters=num_filters, kernel_size=(2*index+4,2*index+4), strides=2, padding='same', activation="relu"))	
	# model.add(layers.Conv2D(filters=15, kernel_size=(10,10), strides=1, padding='same', activation="relu")) #original configuration
	model.add(layers.Flatten())
	model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.01)))
	model.add(layers.Activation('softmax'))
	model.compile(loss='squared_hinge',
	              optimizer='adadelta',
	              metrics=['accuracy'])
	return model

def customSVM():
	model = keras.Sequential()
	model.add(layers.Conv2D(filters=16, kernel_size=(4,4), strides=1,
		padding='same', activation="relu", input_shape=(480,640, 3)))
	model.add(layers.Conv2D(filters=16, kernel_size=(4,4), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=16, kernel_size=(8,8), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=24, kernel_size=(8,8), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=24, kernel_size=(16,16), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=32, kernel_size=(32,32), strides=2, padding='same', activation="relu"))	
	model.add(layers.Conv2D(filters=32, kernel_size=(4,4), strides=2, padding='same', activation="relu"))	
	model.add(layers.Flatten())
	model.add(layers.Dense(10, kernel_regularizer=keras.regularizers.l2(0.01)))
	model.add(layers.Activation('softmax'))
	model.compile(loss='squared_hinge',
	              optimizer='adadelta',
	              metrics=['accuracy'])
	return model

dataLoader = create_dataset.DataLoader()
dataset = dataLoader.getDataset()
#blah = iter(dataset).next()
#model = CreateSVM()
dataset = dataset.shuffle(100).batch(5)
#history = model.fit(dataset, epochs=5)