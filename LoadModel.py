from tensorflow import keras
from image_mod_functions import standardize_image
import functools

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

class ReadyModel():
	"""docstring for ClassName"""
	def __init__(self, model_type="cnn", input_shape=(128,128, 3)):
		model_type = model_type.lower()
		self.gray = False # keeps track of the need to grayscale
		self.input_shape = input_shape
		if model_type == "generalizedsvm":
			self.model = keras.models.load_model('generalizedSVM.h5', custom_objects=dependencies)
		elif model_type == "fastsvm":
			self.model = keras.models.load_model('fastSVM.h5', custom_objects=dependencies)
		else:
			raise Exception(f"Model type {model_type} not found. Try another model.")

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
		"""Returns the top 'num_classes' from model predictions"""
		if len(input.shape) == 4: #handle batch prediction
			classes = []
			results = self.model.predict(input)
			for prediction in results:
				classes.append(self.MaxList(prediction, num_classes))
			return classes
		elif len(input.shape) == 3: #handle single prediction
			result = self.model.predict(input.reshape(1, *input.shape))
			return self.MaxList(result[0], num_classes)
			

if __name__ == '__main__':
	import create_dataset
	import matplotlib.pyplot as plt
	model = ReadyModel("generalizedSVM")
	dataLoader = create_dataset.DataLoader()
	dataset = dataLoader.getDataset(start_index=300)[0]
	
	#single prediction example
	sample = iter(dataset).next()[0].numpy() #pulls one sample image
	singleResult = model.classify(sample)
	print(singleResult)
	fig, ax = plt.subplots(2,1)
	fig.suptitle("Single Sample")
	ax[0].imshow(sample)
	for subIndex, emote in enumerate(singleResult):
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
