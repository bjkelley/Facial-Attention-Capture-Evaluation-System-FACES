# ECS_171_ML_Measure_Attentiveness

## Dependencies
* Python 3.75+
* Numpy 1.17.4
* Scipy 1.3.2
* matplotlib
* Pillow 6.2.1
* Tensorflow 2.0+
* opencv-python (4.1.1.26) 
* Virtualenv

## Modules

### create_dataset.py
The focus of this module is to create a tensorflow dataset object of labeled images from a folder organization. The "getDataset" function does exactly this by globbing through the folder hierarchy in sorted order and returning the dataset. It is important to note that the "getDataset" function as well normalizes the images to float on [0,1] and converts them to greyscale. Ultimately the "getDataset" functions returns this normalized, greyscale, tensorflow dataset object of the images acquired from the FacesDB dataset used for this project.

### image_mod_functions.py
blurb about most prominent functions to use

### loader_class.py
To load a pretrained model, create a new **ReadyModel** object with the name of the model you wish to use ("cnn2" is loaded by default). To make a prediction, call ```results = ReadyModel.classify(batch, k_most_confident_classes=3)``` where batch is either a numpy array with images of shape ```(batch_size, height, width, num_channels)``` or simply  ```(height, width, num_channels)```. ```results``` will contain list where each element is a list with length ```k_most_confident_classes``` and whose inner elements are tuples of ```(probability, emotion_string)```

*```ReadyModel.preprocess()``` is called by ```ReadyModel.classify()```, and expects a square-image input with a face already isolated. It may convert the image to grayscale, and resizes as neccessary for the model_type selected (the most accurate CNN model, 'cnn2', will grayscale and resize inputs to (60, 60, 1)).*

### train_emotion_classifier.py

- this module trains and outputs a keras model (weights, optimizer state, and architecture), as well as provides evaluation plots (loss, accuracy, and top3 accuracy for train/test splits).

- the general steps of this data pipeline are as follows:
    1) **load in the image dataset with Dataloader():**
        - modified the class to return a tensorflow dataset, as well as numpy array of features and labels
        - also included a standardize_image() function to be able to downsize an image
        - also included a one-hot-encoding map to map the emotion strings to one-hot arrays, and have a reverse lookup dictionary as well
    2) **preprocess the images with apply_transformations():**
        - use OpenCV to detect and crop face in the image (the images from the dataset weren't fully cropped)
        - change to grayscale (done in the OpenCV face detection step)
        - downsize all images (used 60x60 pixels) for input to a convolutional neural network
        - apply rest of transformations (add gaussian noise, random rotation)
    3) **create training/testing split, and bootstrap dataset:
        - our data set only had 36 unique faces, and 10 emotions per each face, so 360 faces total. to create an accurate training pipeline, we split up the training data to have 28 of the faces (280 images), then bootstrapped this training set by applying image augmentations 4 more times, resulting in 1,400 images in the training set. the validation set had 80 images, and none of the people in the testing set were present in the training set. 
        - this decreased accuracy in our validation pipeline, but helps us generalize our model to new people
    4) **train convolutional neural network:**
        - using tensorflow's keras API, we trained a CNN (also tried numerous SVM models) over epochs and kept track of the following metrics over the training epochs:
            - **train/test accuracy**
            - **train/test top3 accuracy** (proportion of predictions where true label is in the top 3 predicted classes with highest probability)
            - **train/test loss**
     5) **plotting evaluation figures:**
        - evaluation figures can be found in figures/ directory
        - model CNN2 performed best (as of now) and should be used for further testing
        
     6) **saving tensorflow.keras model):**
        - saved model into models/ directory, using **model.save(*path*)**
        - using keras model.load_model(*path*), you can load in the exact state of the saved model, which includes the current optimizer state, the weights, and the architecture. 
        
### loader_class.py
- this class loads in a specified, pretrained model. by default, loads in the CNN2, trained with 60x60 grayscale face inputs.
- this class **preprocesses** and **classifies**. the classification returns a tuple of 3 elements: the most confident emotion string prediction, the top k (3 by default) most confident emotion string predictions, and the corresponding top k (3 by default) most confident emotion string predictions.
       


## Installation
This install process assumes you have already installed python**3.7.5**. Visit [here](https://www.python.org/downloads/release/python-375/) to download. Virtualenv can also be installed with ```pip install virtualenv```.
### Step 1
Clone our repository using:

    git clone https://github.com/bjkelley/ECS_171_ML_Measure_Attentiveness.git

### Step 2
Make a new directory to contain the project files; create a virtual environment to manage installed packages and activate it with:

    virtualenv -p python3.7 venv
    source venv/bin/activate
for linux or mac, or

    virtualenv -p python3.7 venv
    venv/Scripts/activate
for windows.

### Step 3
Download dependencies using:

    pip install -r requirements.txt
