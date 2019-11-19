from loader_class import *
import matplotlib.pyplot as plt
import numpy as np

# declare model loader to read in pretrained CNN
myLoader = LoadModel(model_type='cnn2', input_shape=(60,60,1))

# read in example image to be classified
image = plt.imread('facesdb/s003/bmp/s003-00_img.bmp')
y_true = 'neutral frontal'

plt.imshow(np.squeeze(image))
plt.title('raw image')
plt.show() 

# do enough preprocessing for model input(facial detection, grayscale, pixel resizing)
image = myLoader.preprocess(image)

plt.imshow(np.squeeze(image))
plt.title('after preprocessing')
plt.show() 

# classify image (does preprocessing for it)
top_class, top_k_classes, top_k_probs = myLoader.classify(image, k_most_confident_classes=3)

print(f"true class: {y_true}")
print(f"predicted class: {top_class}")
print(f"predicted top 3 classes (sorted): {top_k_classes}")