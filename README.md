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

### emotion_classification.py


- this module trains and outputs a keras model (weights, optimizer state, and architecture), as well as provides evaluation plots (loss, accuracy, and top3 accuracy for train/test splits).

- the general steps of this data pipeline are as follows:
    1) load in the image dataset with Dataloader()
        - modified the class to return a tensorflow dataset, as well as numpy array of features and labels
        - also included a standardize_image() function to be able to downsize an image
    2) preprocess the images by appling several transformations
        - **in apply_transformations()**:
            - use OpenCV to detect and crop face in the image (the images from the dataset weren't fully cropped)
            - change to grayscale (done in the OpenCV face detection step)
            - use downsize all images (used 60x60 pixels)


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
