from tensorflow import keras
from image_mod_functions import standardize_image, rgb2gray
import functools
import numpy as np
import time

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

class ReadyModel():
	'''
	ReadyModel: class to load pretrained image emotion classifier
	contains functions to:
	- return predicted emotion string
	- preprocess an image to get it ready for classifiaction
	- return top k predictions
	'''
	def __init__(self, model_type="cnn2"):
		#initialize class members
		self.model_type = model_type.lower()
		self.input_shape = (60,60, 1)
		# Set variables to indicate neccessary preprocessing steps
		self.gray = False 
		self.normalize = False 

		if self.model_type == "cnn2":
			self.gray = True
			self.model = keras.models.load_model('trained_models/cnn2.h5', custom_objects=dependencies)
		elif self.model_type == "cnn1":
			self.gray = True
			self.model = keras.models.load_model('trained_models/cnn1.h5', custom_objects=dependencies)
		elif self.model_type == "generalizedsvm":
			self.input_shape=(128,128,3)
			self.normalize = True
			self.model = keras.models.load_model('trained_models/generalizedSVM.h5', custom_objects=dependencies)
		elif self.model_type == "fastsvm":
			self.normalize = True
			self.input_shape=(128,128,3)
			self.model = keras.models.load_model('trained_models/fastSVM.h5', custom_objects=dependencies)
		else:
			raise Exception(f"Model type {model_type} not found. Try another model.")

	def Preprocess(self, batch):
		'''preprocesses batch according to model-specific requirements
		'''
		if len(batch.shape) == 3: #group single input as batch
			batch = batch.reshape(1, *batch.shape)
		readyBatch = np.empty(shape=(batch.shape[0],*self.input_shape))
		for index, image in enumerate(batch):
			# grayscale and standardize first for maximum precision
			if self.gray:
				image = rgb2gray(image, self.input_shape[2])
			# scale image to input_shape width x height
			image = standardize_image(image, self.input_shape[0], self.input_shape[1])
			# normalize if necessary
			if self.normalize:
				image = image / 255.0
			readyBatch[index]= image #store the batch in new array
		return readyBatch

	def MaxList(self, result, k_most_confident_classes):
		""" Pulls the top 'k_most_confident_classes' emotions from result and returns a list of tuples
			containing """
		classList = []
		for index, probability in enumerate(result):
			isMax = False
			#iterate through sorted subset of result
			for rank, element in enumerate(classList):
				if probability > element[0]:
					classList.insert(rank, (probability, self.GetEmotion(index)))
					isMax = True
					break
			if len(classList) < k_most_confident_classes and not isMax: #fills the list with values
				classList.append((probability, self.GetEmotion(index)))
				continue
			if len(classList) > k_most_confident_classes: #keeps only top k_most_confident_classes in result
				classList.pop()
		return classList

	def GetEmotion(self, x):
		"""Decodes emotions based on an index from 0-9"""
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

	def classify(self, input, k_most_confident_classes=3):
		"""
		Returns a list with the top 'k_most_confident_classes' from model prediction.
		Each element is a tuple of (probability, emotion)
		"""
		batch = self.Preprocess(input) #handle preprocessing before model input
		classes = []
		results = self.model.predict(batch)
		for prediction in results:
			classes.append(self.MaxList(prediction, k_most_confident_classes))
		return classes
			

if __name__ == '__main__':
	print("Loading Model...")
	model = ReadyModel()
	print("Importing modules for testing...")
	import create_dataset
	import matplotlib.pyplot as plt
	# import time
	print("Loading data...\r")
	startTime = time.time()
	dataLoader = create_dataset.DataLoader()
	dataset = dataLoader.getDataset(start_index=300)[0]
	loadTime = time.time() - startTime
	print(f"Loaded in {loadTime} seconds.")
	
	#single prediction example
	print("Loading single sample...", end="\r", flush=True)
	sample = iter(dataset).next()[0].numpy() #pulls one sample image
	print("Making single prediction...", end="\r", flush=True)
	startTime = time.time()
	singleResult = model.classify(sample)
	predictTime = time.time() - startTime
	print(f"Processed in {predictTime} seconds.")
	fig, ax = plt.subplots(2,1)
	fig.suptitle("Single Sample")
	ax[0].imshow(sample/255.0) # normalzie to prevent clipping
	for subIndex, emote in enumerate(singleResult[0]):
		ax[1].bar(emote[1], emote[0]*100)
	plt.show()

	#batch prediction example
	print("Loading batch of sample...", end="\r", flush=True)
	batchSample = iter(dataset.shuffle(10).batch(20)).next()[0].numpy() #pulls 5 sample images
	print("Making batch prediction...", end="\r", flush=True)
	startTime = time.time()
	results = model.classify(batchSample)
	batchPredictTime = time.time() - startTime
	print(f"Processed batch in {batchPredictTime} seconds.")
	for index, result in enumerate(results):
		fig, ax = plt.subplots(2,1)
		ax[0].imshow(batchSample[index]/255) # normalize to prevent clipping
		plt.suptitle(f"Image {index}")
		for subIndex, emote in enumerate(result):
			ax[1].bar(emote[1], emote[0]*100)
		plt.show()
