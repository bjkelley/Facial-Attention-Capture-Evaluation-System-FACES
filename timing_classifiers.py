from tensorflow import keras
from image_mod_functions import standardize_image, rgb2gray
import functools
import numpy as np
import time
import create_dataset

# define custom metric, needed as a dependency in keras.models.load_model
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
dependencies = {'top3_acc' : top3_acc}

dataLoader = create_dataset.DataLoader()
dataset = dataLoader.getDataset()[0]
dataset = dataset.shuffle(100)


def test_model(model_name, model_type, layers, drop, lr, reg, batch):
	current_model = keras.models.load_model(f"trained_models/{model_name}", custom_objects=dependencies)
	current_set = dataset.batch(1)
	for current_batch in current_set:
		startTime = time.time()
		results = current_model.predict(current_batch)
		batchPredictTime = time.time() - startTime


