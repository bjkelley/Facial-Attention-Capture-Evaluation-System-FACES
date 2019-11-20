# ECS_171_ML_Measure_Attentiveness

## Dependencies
* Python 3.75+
* Numpy 1.17.4
* Scipy 1.3.2
* Pillow 6.2.1
* Tensorflow 2.0+
* opencv-python (4.1.1.26) 
* Virtualenv
* matplotlib
* etc

## Modules

### create_dataset.py
blurb about most prominent functions to use

### image_mod_functions
blurb about most prominent functions to use

### how to actually use the CNN classifier (for live demo stuff):
- to actually use the classifier for working on the live demo part, **predict_new_sample_example.py** is really helpful. you basically need to load the CNN classifer with the custom class **LoadModel**, read in the image, call ```image = LoadModel.preprocess(image)``` (, then call ```pred, top3_preds, top3_probs = LoadModel.classify(image)```. It might be noteworthy that ```LoadModel.preprocess()``` actually does facial detection/cropping with OpenCV, converts the image to grayscale, and resizes to the correct pixel amount needed (60x60 for the most accurate CNN model, 'cnn2').
- **output of LoadModel.classify(image)**
    - a tuple of 3 elements:
        - predicted class: (ex neutral frontal)
        - predicted top 3 classes (sorted): (ex ['neutral frontal', 'closed', 'sadness'])
        - predicted top 3 classes probs (sorted): (ex [0.53434867, 0.26075634, 0.16289178])
- it's pretty helpful to open a jupyter notebook, paste all the code in **predict_new_sample_example.py** into a cell, and then looking at the output. or also can just run ```python predict_new_sample_example.py``` in a terminal.
- i'm assuming for the live demo, we will be extracting each frame and classifying the emotion in that frame. we should probably do a very low frame rate (maybe like 3 per second?) and make sure that in the live demo we hold the emotions for a second lol. 

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
**Fill this out as you create modules** to let everyone know how to setup important directories and how to download the right files for use.
### Step 1
Make a new directory to contain the project files; create a virtual environment to manage installed packages and activate it:

    virtualenv -p python3.7 venv
    source venv/bin/activate
    
or for windows:

    virtualenv -p python3.7 venv
    venv/Scripts/activate
    
or for mac:

    virtualenv -p python3.7 venv
    source venv/bin/activate
