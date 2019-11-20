from tensorflow import keras
from image_mod_functions import standardize_image, rgb2gray
import functools

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

class ReadyModel():
	'''
    LoadModel: class to load pretrained image emotion classifier
    contains functions to:
        - return predicted emotion string
        - preprocess an image to get it ready for classifiaction
        - return top prediction and/or top3 predictions
    '''
	def __init__(self, model_type="cnn2"):
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
			self.model = keras.models.load_model('generalizedSVM.h5', custom_objects=dependencies)
		elif self.model_type == "fastsvm":
			self.normalize = True
			self.input_shape=(128,128,3)
			self.model = keras.models.load_model('fastSVM.h5', custom_objects=dependencies)
		else:
			raise Exception(f"Model type {model_type} not found. Try another model.")

	def Preprocess(self, batch):
		'''preprocesses batch accordingly so that model can take it as input
		- use OpenCV face detector and convert to grayscale
		- resize image
		'''
        # if type(bath) 
		if len(batch.shape) == 3: #group single input as batch
			batch = batch.reshape(1, *batch.shape)
		for index, image in enumerate(batch):
			# grayscale and standardize first for maximum precision
			if self.gray:
				image = rgb2gray(image, self.input_shape[2])
			# scale image to input_shape width x height
			image = standardize_image(image, self.input_shape[0], self.input_shape[1])
			# normalize if necessary
			if self.normalize:
				image = image / 255
			batch[index] = image #store the result in original numpy array
		return batch

	def MaxList(self, result, num_classes):
		""" Pulls the top 'num_classes' emotions from result and returns a list of tuples
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
			if len(classList) < num_classes and not isMax: #fills the list with values
				classList.append((probability, self.GetEmotion(index)))
				continue
			if len(classList) > num_classes: #keeps only top num_classes in result
				classList.pop()
		return classList

	def GetEmotion(self, x):
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

	def classify(self, input, num_classes=3):
		"""
		Returns a list with the top 'num_classes' from model prediction.
		Each element is a tuple of (probability, emotion)
		"""
		batch = self.Preprocess(input) #handle preprocessing before model input
		classes = []
		results = self.model.predict(batch)
		for prediction in results:
			classes.append(self.MaxList(prediction, num_classes))
		return classes
			

if __name__ == '__main__':
	import create_dataset
	import matplotlib.pyplot as plt
	model = ReadyModel("generalizedSVM")
	dataLoader = create_dataset.DataLoader()
	dataset = dataLoader.getDataset(start_index=300)[0]
	
	#single prediction example
	sample = iter(dataset).next()[0].numpy() #pulls one sample image
	singleResult = model.classify(sample)
	fig, ax = plt.subplots(2,1)
	fig.suptitle("Single Sample")
	ax[0].imshow(sample)
	for subIndex, emote in enumerate(singleResult[0]):
		ax[1].bar(emote[1], emote[0]*100)
	plt.show()
	
	#batch prediction example
	batchSample = iter(dataset.shuffle(10).batch(20)).next()[0].numpy() #pulls 5 sample images
	results = model.classify(batchSample)
	for index, result in enumerate(results):
		fig, ax = plt.subplots(2,1)
		ax[0].imshow(batchSample[index])
		plt.suptitle(f"Image {index}")
		for subIndex, emote in enumerate(result):
			ax[1].bar(emote[1], emote[0]*100)
		plt.show()
