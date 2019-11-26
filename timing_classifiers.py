from tensorflow import keras
from image_mod_functions import standardize_image, rgb2gray
import functools
import numpy as np
import time
import create_dataset
import classifier_definitions

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

dataLoader = create_dataset.DataLoader()
dataset = dataLoader.getDataset()[0]
eval_dataset = dataLoader.getDataset(start_index=300)[0]

dataset = dataset.shuffle(100)


def test_model(model_name, model_params, test_batch_size):
	model_type = model_params[0]
	layers = model_params[1]
	drop = model_params[2]
	lr = model_params[3]
	reg = model_params[4]
	batch = model_params[5]
	current_model = keras.models.load_model(f"trained_models/{model_name}", custom_objects=dependencies)
	timings = test_batch_size
	j = 0

	current_set = eval_dataset.batch(10)
	accuracy = current_model.evaluate(current_set)
	print(accuracy)

	for b in test_batch_size:
		current_set = dataset.batch(b)
		print(f"Timing batches of size {b} for {model_name}")

		i = 0
		average = 0

		for current_batch in current_set:
			startTime = time.time()
			current_model.predict(current_batch)
			batchPredictTime = time.time() - startTime
			if i == 0:
				average = batchPredictTime
			else:
				total = (average * i) + batchPredictTime
				average = total / (i + 1)

			i = i + 1
		timings[j] = average
		j = j + 1

	# print(f"Times: {timings}")
	return np.concatenate((model_params, timings, accuracy[1:]), axis=None)


svm1 = test_model("SVM-0#l2-d0.2-lr0.001-b60.h5", ['SVM', 3, 0.2, 0.001, 0.1, 30], [1, 5, 10, 20])
svm2 = test_model("SVM-1#l2-d0.4-lr0.001-b60.h5", ['SVM', 3, 0.4, 0.001, 0.01, 30], [1, 5, 10, 20])
svm3 = test_model("SVM-2#l2-d0.3-lr0.001-b60.h5", ['SVM', 3, 0.5, 0.001, 0.01, 60], [1, 5, 10, 20])
svm4 = test_model("SVM-3#l2-d0.5-lr0.001-b60.h5", ['SVM', 3, 0.3, 0.001, 0.1, 60], [1, 5, 10, 20])

print(svm1)
print(svm2)
print(svm3)
print(svm4)

# dataset = classifier_definitions.reduceSize(dataset)
# eval_dataset = classifier_definitions.reduceSize(eval_dataset)
#
# cnn1 = test_model("CNN-0#l2-d0.2-lr0.001-b60.h5", ['CNN', 2, 0.2, 0.001, 'na', 60], [1, 5, 10, 20])
# cnn2 = test_model("CNN-1#l2-d0.4-lr0.001-b60.h5", ['CNN', 2, 0.4, 0.001, 'na', 60], [1, 5, 10, 20])
# cnn3 = test_model("CNN-2#l2-d0.3-lr0.001-b60.h5", ['CNN', 2, 0.3, 0.001, 'na', 60], [1, 5, 10, 20])
# cnn4 = test_model("CNN-3#l2-d0.5-lr0.001-b60.h5", ['CNN', 2, 0.5, 0.001, 'na', 60], [1, 5, 10, 20])
#
# print(cnn1)
# print(cnn2)
# print(cnn3)
# print(cnn4)



