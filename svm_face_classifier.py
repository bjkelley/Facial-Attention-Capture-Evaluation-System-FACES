import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import create_dataset
import functools
import numpy as np
import matplotlib.pyplot as plt

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

def customSVM(input_shape = (480,640, 3)):
	model = keras.Sequential()
	model.add(layers.Conv2D(filters=16, kernel_size=(4,4), strides=1,
		padding='same', activation="relu", input_shape=input_shape))
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

dataLoader = create_dataset.DataLoader()
trainSet = dataLoader.getDataset(num_samples=300)[0] #gives first 30 subjects
testSet = dataLoader.getDataset(start_index=300)[0] #gives last 6 subjects
testSet = testSet.batch(10)

#Apply transformations to expand dataset
# trainSet = trainSet.repeat(3)
trainSet = applyTransform(trainSet, num_repeat=5)
trainSet = trainSet.shuffle(200).batch(15)

# sample = iter(trainSet).next() #first batch
# for image in sample[0]:
# 	plt.imshow(image)
# 	plt.show()

model = fastSVM()
model.summary()
history = model.fit(trainSet, epochs=250, validation_data=testSet)

plt.plot(history.history['accuracy'], label="train acc")
plt.plot(history.history['val_accuracy'], label="test acc")
plt.plot(history.history['top3_acc'], label="train top3")
plt.plot(history.history['val_top3_acc'], label="test top3")
plt.title("fastSVM repeat=5")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
