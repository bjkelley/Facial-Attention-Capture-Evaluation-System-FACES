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
A suite of functions to manipulate numpy image matrices. Manipulations include rotation, additive gaussian noise, speckle, up and down sampling, size standardization, to gray scale.

### loader_class.py
To load a pretrained model, create a new **ReadyModel** object with the name of the model you wish to use. To make a prediction, call ```results = ReadyModel.classify(batch, k_most_confident_classes=3)``` where batch is either a numpy array with images of shape ```(batch_size, height, width, num_channels)``` or simply  ```(height, width, num_channels)```. ```results``` will contain list where each element is a list with length ```k_most_confident_classes``` and whose inner elements are tuples of ```(probability, emotion_string)```. The default behavior uses the "cnn2" model and returns the 3 most confident classes.

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


### classifier_definitions.py
Use the `get_CNN()` and `get_generalizedSVM()` functions to define models of each architecture with specified parameters.
These functions make it easier to perform the Grid-Search and build the best models once determined.
`get_generalizedSVM()` takes in the following parameters: `input_shape`, `num_conv_layers` determines the number of
 hidden convolutional layers, `dropout` gives the dropout rate after each convolutional layer, `learning_rate`, and
 `regularizer` gives the regularizer hyperparameter. `get_CNN()` takes in the following parameters: `num_conv_layers`
 determines the number of hidden convolutional layer groups (each group has three convolutional layers and a dropout),
  `dropout` gives the dropout rate after each convolutional layer,
 and `learning_rate`.

### grid_search.py
Grid-Searches can be perfomed by defining a list of values for each hyperparameter; the lists can be of different sizes.
a Grid-Search of a single model type (either 'SVM' or 'CNN') is performed by calling
`run_grid_search(model_type, layer_options, dropout_options, lr_options, reg_options, batch_options, skip)` where each variation
of hyperparameters is tested, with the exception of the first `skip` number of models. For each model, the test and train accuracy
and top three accuracy is saved on a graph and saved in the current folder.

Once the grid search is complete, you can save and compare the results of some of the best performing models. Define a
list of the same size for each hyperparameter, representing the values for that parameter of the n-best models to be
compared. Call the evaluation function as follows:
`evaluate_best_models(model_type, n_best, layer_options, dropout_options, lr_options, reg_options, batch_options, epochs=200, patience=20)`
We recommend that you use a large number of epochs so that early-stopping is likely to occur. This ensures that the most
accurate weights are saved, not just the final weights. All models included will be trained, stored in the trained models folder,
and have their history of test accuracy stored on one graph for comparison.

### evaluate_classifiers.py
The function test_model returns average times for prediction of each batch size in the `test_batch_size` list. It also determines accuracy
on the test set - the 6 people left out of the train set. Just pass the model parameters in as a list, and provide the
filename of the pre-trained saved model: `test_model(model_name, model_params, test_batch_size)`.

`eval_best_models()` is a predefined function that calls `test_model()` on each of the top eight models found in our
grid search.

`sample_best_models()` produces sample output of the top 2 models - the best SVM and CNN. It produces a graph with
several images and the top three predictions from each model.
![alt text](/figures/svm_samples_4.png =250x)
![alt text](/figures/cnn_samples_4.png =250x)



 ### SegmentorClass.py

 Contains driver code for each of the three models we used. Each model has a specific `init` function and `segment`
 function to return the bounding box coordinates along with an array of the resized faces to feed into the emotional classification models.
 The model functions are predetermined when initializing the `Segmentor` class where it takes in the parameter of `impl` which you can define to be one of the three models "Haar", "Yolo", or "faced". You can then use the `Segmentor` instance to call the `segment(frame)` method which takes in a numpy array representing the image to be processed.


## Installation
*Note:* This install process assumes you have already installed python**3.7.5**. Visit [here](https://www.python.org/downloads/release/python-375/) to download. Virtualenv can also be installed with ```pip install virtualenv```.
### Step 1
Clone our repository using:

    git clone https://github.com/bjkelley/ECS_171_ML_Measure_Attentiveness.git

### Step 2
Make a new directory to contain the project files; create a virtual environment to manage installed packages and activate it.

for linux or mac:

    virtualenv -p python3.7 venv
    source venv/bin/activate

or for windows:

    virtualenv -p python3.7 venv
    venv/Scripts/activate

*troubleshooting:* Machines with multiple versions of python will have multiple names for each version. If ```-p python3.7``` doesn't work, you can replace with ```-p python3``` or even ```-p python```. Once your virtual environment is active, type ```python --version``` to ensure that you are using python 3.7.5

### Step 3
Download dependencies using:

    pip install -r requirements.txt
