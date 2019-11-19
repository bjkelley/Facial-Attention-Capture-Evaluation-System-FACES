from loader_class import *
import matplotlib.pyplot as plt
import numpy as np

# declare model loader to read in pretrained CNN
myLoader = LoadModel(model_type='cnn2', input_shape=(60,60,1))

# read in example image to be classified
image = plt.imread('facesdb/s001/bmp/s001-00_img.bmp')
y_true = 'neutral frontal'

# classify image (does preprocessing for it)
top_class, top_k_classes, top_k_probs = myLoader.classify(image, k_most_confident_classes=3)

plt.imshow(np.squeeze(image))
plt.show()

print(f"true class: {y_true}")
print(f"predicted class: {top_class}")
print(f"predicted top 3 classes (sorted): {top_k_classes}")