---
layout: post
title:      "Identifying Dog Breeds with Convolutional Neural Networks"
date:       2020-02-10 05:09:36 +0000
permalink:  identifying_dog_breeds_with_convolutional_neural_networks
---

### Motivation.
The task of correctly identifying a dog's breed is a challenge for most people. Some breeds differ only slighlty, appearing identical except for subtle fur or shape distinctions. Other breeds have a wide range of fur colors and types which increases the complexity of making the right determination. Due to the high inter-class and intra-class variation, this use case is ideal for practicing deep learning skills. This article will compare self-built (from scratch) convolutional neural network architectures and contrast the results with a transfer learning model built from pre-trained layers. 

### Project details. 
* **Goal:** This investigation will walk through iterations of the process of building a convolutional neural network (CNN) to classify dog breeds from images.

* **Method:**  Self-built and pre-trained CNNs will be compared to identify the best model. 

* **Dataset:** 20,580 images of dogs from 120 breeds. The images display variation in background setting, subject position & orientation, and image quality. The dataset is sourced from the [kaggle](https://www.kaggle.com/) machine learning datasets, located [here](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset).

* **Development Environment:** Since the ability to work with neural networks is highly dependent on machine specifications (e.g., memory, processor, operating system, video card), this project will use cloud services from Googe [Colab](https://colab.research.google.com/). Colab allows scripting in Jupyter Notebook with Python and provides access to data in Google Drive cloud storage.

Before moving on to designing and building models, I'll define the ideas relevant to this project and explain how they can affect results.

### Concepts and terminology. 

* **Convolutional Neural Networks (CNN):** The type of neural network that performs best will depend on the situation in which it is used. Outcome precision, network size, and execution speed are all examples of variables influenced by the choice of network. CNNs are a deep network used for image recognition tasks in which convolutions over the input layer are used to compute the model output. This results in local connections where each region of the input is eventually connected to a neuron in the output layer. The hidden layers of this network apply different filters that then combine into the results. CNN filters learn to capture spatial features (e.g., line, curve, edge) from an image based on weights learned through back propagation. Stacked layers of filters are able to detect complex shapes from the features learned at every layer. 

* **Pre-Trained Model:** Constructing a neural network from scratch is a formidable task due to the number of hyperparameters that must be adjusted in order to attain high performance. This is one reason why transfer knowlege is viewed as an attractive alternative. Transfer knowlege uses a neural network that has been trained by others on similar data. Pre-trained models are able to find early stage features like edges, curves, gradients and repeating patterns. In many cases they are able to locate the main features, too. A neural net built from pre-trained models replaces its output layer with new layers that learn how to utilize the pre-trained features and output class probabilities for the particular use case.

* **Forward Propagation:**  The process of taking input, acting on it through hidden layers, and returning an estimate in an output layer. 

* **Backward Propagation:**  The process of minimizing the error contribution of every neuron  in each layer by updating their weights when traveling backward through the network.

* **Hyperparameters:** Configurable values that are set before the learning process begins.  Hyperparameter values dictate how the model learns its parameters from the data (i.e., training algorithm behavior) .

* **Train-Validation-Test Split:** Models are fit on the training set and evaluated on the validation set to tune parameters against validation set accuracy. The final model predicts labels of test set data to provide an unbiased estimate of the fit. Altogether, this process helps protect against overfitting.

* **Overfitting:** Typical models have millions of parameters. When these models are trained for many epochs there is a high probability they will overfit to the training data.  An overfit model does not generalize well to observations in validation and test sets. Overfitting is confirmed if the validation set accuracy is less than training set accuracy.

* **Normalization:** Standardizing inputs decreases training time and reduces the chance of convergence issues. 

* **Convergence Issues:** Issues occurring at saddle points and local minima where the graidents are nearly zero. Adjusting hyperparameters such as the learning rate and changing the optimizer (to Adam or RMSProp) prevents models from falling into these positions and failing to learn.

* **Neural Network Hyperparameters:**
 *  **Number of Layers:** The total count of layers within the neural network. A high number of layers contributes to overfitting and vanishing & exploding gradient problems. A low number of layers contributes to high bias and low accuracy. Optimal values depend on the size of training data.  It is typically advised to add layers until overfitting. At this point, regularization components  (e.g, L1/L2 regularization, dropout, batch normalization) can be used to reduce overfitting.
 *  **Number of Epochs:** The number of times the entire set of training data is shown to the model. Too many epochs can lead to overfitting and failure to generalize to unseen data. Too few epochs can limit model accuracy. A rule of thumb is to have the epoch count somewhere in the range of 20 to 50. 	 
 * **Batch Size:** The number of samples shown to the network before the weights are updated. Low batch sizes have volatile weights and convergence issues. High batch sizes increase computation time. A rule of thumb is to use batch sizes in powers of two (i.e., 32, 64, 128, 256).
 * **Learning Rate:**  The rate at which the search for the optimal solution is made. Low values increase computation time and high values risk failure to converge to minima. A rule of thumb is to try values that are powers of ten when searching for the optimal learning rate (e.g., 0.001,0.01, 0.1). Learning rate values are somewhat dependent on the optimizer algorithm selected.
 * **Activation Function:** Functions that define the output of a layer. Common choices are Sigmoid, Tanh, ReLU and LeakyReLU.
 * **Metric:** The function used to judge the performance of a model. Common choices are Accuracy and F1-score.
 * **Loss:**  The cost function that is minimized to measure how good a model performs in predicting the expected outcome. Common choices are categorical cross-entropy and mean-squared error.
 * **Optimizer:**  The algorithm used by the model to update the weights of each layer after every iteration. Common choices are SGD, RMSProp and Adam.
 * **Initialization:**  Weight adjustments to layer activation outputs that prevent exploding or vanishing gradients during a forward pass through the neural network. Generally, the default selections work well (e.g., He initialization for ReLu activation and Glorot for Sigmoid).	 
 * **Dropout:** A regularizing layer that removes connections between hidden units in every iteration to prevent dependency on any particular feature. Dropout values are set between 0 & 1 and are based on model overfitting.	 
 * **L1/L2 Regularization:** A regularizing paramer that truncates high weight values so that the model is not dependent on any particular feature. Regularization is generally used when the model continues to overfit after implementing high Dropout.

* **CNN Hyperparameters:**
 * **Filter Size:** Filters measure how close a section of the input matches a feature. The weights of a filter matrix are derived from convolutions on the training data. Small filters collect local information while big filters collect more global representations. A general rule of thumb is to use a large filter sizes (e.g., 13 x 13) if many pixels are necessary to recgonize an object. If class differences between objects occupy few pixels then small filter sizes are ideal (e.g., 3 x 3). High performance neural networks usually start with smaller filters to collect local information and then gradually increase the filter size to begin constructing more global features.
 * **Number of Channels:** This is the number of color bands in the input layer and the number of filters used in the convolution operation for the hidden layers.  As the number of channels increases, the number of features learned and the risk to overfitting also increase.  High performance neural networks start with a small number of channels to detect low-level features which are then combined to form complex shapes to distinguish between classes after the number of channels increases.
 * **Padding:** Padding is the addition of columns and rows of zeroes to affect the spatial size of an image after convolution. This action retains border information in the network. Parameters for padding `Same` and `Valid`. The output size from `Same` is equal to the input size. It is used when information in the borders is considered important. The output size from `Valid` is an image shrunk to `CEILING((n + f - 1) / s)`, where `n` is the input dimensions, `f` is the filter size, `s` is the stride length, and `CEILING` is a function rounding decimal results up to the nearest integer.
 * **Pooling:** Layers that reduce the dimensionality of a provided representation. Max-Pooling is the most commonly used pooling layer. Max-Pooling layers typically use low filter sizes (e.g., 3x3) filter sizes and stride (e.g., 2).
 * **Stride:** This is the number of pixels skipped horizontally and vertically after each multiplication of the input weights with the filter matrix (convolution). Stride contributes to decreasing the input image size.
 * **Batch Normalization:** Batch normalization layers standardize the input with respect to later layers. Normalized input can become too big or too small after passing through the intermediate layers in deep neural networks. At distant layers there is an internal shift that impacts the ability for the network to learn. Generally, batch normalization layers are placed after activation functions and before Dropout layers.

Now that the fundamental concepts underpinning this project have been covered, it's time to get to work.

### Solution.
I'll build five different models with a focus on how changes to CNN structure affects overfitting and accuracy. Four models will be built from the data only. The fifth model will use transfer learning from the [Inception-V3](https://cloud.google.com/tpu/docs/inception-v3-advanced) image recognition model. 

But first, the preliminaries.

#### Obtain the data.
Gather the data from its source.

```
# obtain data
import os

os.environ['KAGGLE_USERNAME'] = "xxxxx"
os.environ['KAGGLE_KEY'] = "XXXXX"
!kaggle datasets download -d jessicali9530/stanford-dogs-dataset

> Downloading stanford-dogs-dataset.zip to /content
> 98% 737M/750M [00:12<00:00, 63.4MB/s]
> 100% 750M/750M [00:12<00:00, 62.4MB/s]
```

View the information received.

```
!ls

> sample_data  stanford-dogs-dataset.zip
```

Unzip the folder the data was received in.

```
!unzip stanford-dogs-dataset.zip

> Archive:  stanford-dogs-dataset.zip
>     inflating: annotations/Annotation/n02085620-Chihuahua/n02085620_10074  
>     ...
>     inflating: images/Images/n02116738-African_hunting_dog/n02116738_9924.jpg  
```

View the unzipped data folders.

```
!ls

> annotations  images  sample_data  stanford-dogs-dataset.zip
```

Make folders for the training set, validation set, testing set, and models.

```
!mkdir train valid test models
!ls

> annotations  models	  stanford-dogs-dataset.zip  train
> images	     sample_data  test			     valid
```

Images are located within the file structure `images\Images\[Breed Identifier]\` where `[Breed Identifier]` is a unique string joined by hyphen with the name of the breed. An example dog breed subfolder is `n02085620-Chihuahua`. An example image within that subfolder is `n02085620_7.jpg`. There is no metadata available on the dataset as a whole. Since I want descriptive information on classes and counts, I'll construct some metadata.

Create a single source of label information mapping images with dog breed.

```
# create a dataframe with images and labels
import pandas as pd

data = []
home_dir = '/content/images'
for folder in sorted(os.listdir(home_dir)):
    for subfolder in sorted(os.listdir(home_dir + '/' + folder)):
      for file in sorted(os.listdir(home_dir + '/' + folder + '/' + subfolder)):
        data.append((subfolder[subfolder.find('-')+1:], file))

df_labels = pd.DataFrame(data, columns=['breed', 'imageID'])
print(df_labels)

>                      breed              imageID
> 0                Chihuahua  n02085620_10074.jpg
> 1                Chihuahua  n02085620_10131.jpg
> 2                Chihuahua  n02085620_10621.jpg
> 3                Chihuahua   n02085620_1073.jpg
> 4                Chihuahua  n02085620_10976.jpg
> ...                    ...                  ...
> 20575  African_hunting_dog   n02116738_9798.jpg
> 20576  African_hunting_dog   n02116738_9818.jpg
> 20577  African_hunting_dog   n02116738_9829.jpg
> 20578  African_hunting_dog   n02116738_9844.jpg
> 20579  African_hunting_dog   n02116738_9924.jpg
```

This step confirms that there are 20,580 images of dogs in the dataset.

View image counts by breed to get a sense of the size of each category. 

```
# view image counts by dog breed
df_labels.groupby('breed').count().sort_values(by='imageID', ascending=False)
```

![Images by Breed](https://github.com/monstott/Blogs/raw/master/Blog5/ImagesByBreed2.PNG)

The breed with the largest number of images available is the Maltese Dog with 252. The breed with the lowest number of images is the Redbone Coonhound at 148. Breeds with the most images have more than 1.5 times as many images as  the breeds with the fewest. This variation in class sizes may affect results since models cannot be trained on equal class samples. This discrepancy is an opportunity for improvement.

The next step is to separate images into groups. 

#### Split into sets.
Create a helper function to count the number of images in a directory tree. This will be used to track the images moved to the `train`, `valid` and `test` folders.

```
# count the files (without the folders) in a directory hierarchy
def fileCount(folder):
 
    count = 0
    for f in os.listdir(folder):
        path = os.path.join(folder, f)

        if os.path.isfile(path):
            count += 1

        elif os.path.isdir(path):
            count += fileCount(path)

    return count
```

Split off the testing set by moving 10% of the images in each breed subfolder to the `test` folder location.

```
# create test set
import shutil

home_dir = '/content/images'
for folder in sorted(os.listdir(home_dir)):
    for subfolder in sorted(os.listdir(home_dir + '/' + folder)):
		
      source = home_dir + '/' + folder + '/' + subfolder + '/'
      destination = '/content/test/' + subfolder[subfolder.find('-')+1:] + '/'
      if not os.path.exists(destination):
        os.makedirs(destination)      
				
      files = os.listdir(source)
      for k, f in enumerate(files):
        if not k % 10:
          shutil.move(source+f, destination)

print('Number of Testing Set Images:', fileCount('/content/test/'))

> Number of Testing Set Images: 2110
```

2,110 images have been moved to the `test` folder. 

Perform the same operation for the validation set. 

```
# create validation set
home_dir = '/content/images'
for folder in sorted(os.listdir(home_dir)):
    for subfolder in sorted(os.listdir(home_dir + '/' + folder)):
		
      source = home_dir + '/' + folder + '/' + subfolder + '/'
      destination = '/content/valid/' + subfolder[subfolder.find('-')+1:] + '/'
      if not os.path.exists(destination):
        os.makedirs(destination)      
				
      files = os.listdir(source)
      for k, f in enumerate(files):
        if not k % 10:
          shutil.move(source+f, destination)

print('Number of Validation Set Images:', fileCount('/content/valid/'))

> Number of Validation Set Images: 1903
```

1,903 images have been moved to the `valid` folder.

Complete the splitting process by placing the remaining files in a training set.

```
# create training set
home_dir = '/content/images'
for folder in sorted(os.listdir(home_dir)):
    for subfolder in sorted(os.listdir(home_dir + '/' + folder)):
		
      source = home_dir + '/' + folder + '/' + subfolder + '/'
      destination = '/content/train/' + subfolder[subfolder.find('-')+1:] + '/'
      if not os.path.exists(destination):
        os.makedirs(destination)      
				
      files = os.listdir(source)
      for k, f in enumerate(files):
          shutil.move(source+f, destination)
					
print('Number of Training Set Images:', fileCount('/content/train/'))

> Number of Training Set Images: 16567
```

16,567 images have been moved to the `train` folder.

Variables for each set can be created now that training, validation, and testing set images have been physically separated. 

Create the set variables. 

```
# create train-validate-test set variables
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
import numpy as np

# function to load data into train-validate-test sets
def splitSets(path, cats=120):
    folder_files = load_files(path)
    file_names = np.array(folder_files['filenames'])
    target_labels = np_utils.to_categorical(np.array(folder_files['target']), cats)
    
    return file_names, target_labels

# apply variable split to the data sets
train_files, train_labels = splitSets('/content/train/')
valid_files, valid_labels = splitSets('/content/valid/')
test_files, test_labels = splitSets('/content/test/')
```

Neural networks work on arrays and not images. The sets of images need to transform their format in order to be used as input.

Define a function to return a row, column and channel 3-dimensional array for an image from its path.

```
# function to return each image as a 4D tensor with dimensions (1, height, width, channel)
from keras.preprocessing import image              

def imagesTo4D(path, dim=224):
    img = image.load_img(path, target_size=(dim, dim)) # resize image to square
    x = image.img_to_array(img) # 3 dimensions

    return np.expand_dims(x, axis=0) # 4 dimensions 
```

Define a function to aggregate the 3-dimensional arrays into a 4-dimensional tensor where the new dimension is the sample image count.

```
# function to return the aggregation of 4D tensors with dimensions (image count, height, width, channel)
from tqdm import tqdm

def stack4DImages(paths):
    image_arrays = [imagesTo4D(path) for path in tqdm(paths)]

    return np.vstack(image_arrays)
```

Apply the array transformation functions to the split sets and normalize the results.

```
# normalize the image arrays to a 0 - 1 scale
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

train_tensors = stack4DImages(train_files).astype('float32') / 255
valid_tensors = stack4DImages(valid_files).astype('float32') / 255
test_tensors = stack4DImages(test_files).astype('float32') / 255

> 100%|██████████| 16567/16567 [01:11<00:00, 230.97it/s]
> 100%|██████████| 1903/1903 [00:09<00:00, 201.82it/s]
> 100%|██████████| 2110/2110 [00:10<00:00, 206.22it/s]
```

With the preliminary steps finished, I can move on to model building. 

#### 1st Model: Build a CNN with repeating convolution and pooling layers.

**Model 1 Details:**

* The first model has 3 convolutional layers with low **filter size** (2 x 2), default **stride** (1 x 1), and `relu` as their **activation function**. The rectified linear (ReLU) activation function is a piecewise linear function that outputs its input if it is positive and zero, otherwise. 
* The **padding** type used in each convolution layer is `Valid` , meaning border information is not preserved and image size will decrease. 
* Each convolution layer has a **max-pooling layer** with a low filter size (2 x 2) following it to decrease the number of parameters (and dimensionality). 
* The **number of layers** in the convolution layers increases with network depth (16, 32, 64). 
* At the end of the network, a **global average pooling layer** is used to severly reduce the number of parameters and then the `softmax` (Softmax) **activation function**  is applied to normalize input into a probability distribution of class probabilities. 
* The **optimizer** is set to `adam` (Adam). The Adam optimization algorithm uses the power of adaptive learning rate methods to find individual learning rates for each parameter. 
* The **loss** is set to `categorical_crossentropy` (Categorical Cross-Entropy), a function intended for multi-class classification where each sample input belongs to a single class. 
* The performance **metric** is set to `accuracy` (Accuracy), the ratio of correct predictions out of total predictions.
* The fit will use an `input_shape` (**Image Size**) of (224, 224, 3), 10 `epochs` (**Epochs**) and a `batch_size` (**Batch Size**) of 32.

Construct the first model architecture.

```
# first model architecture
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, concatenate
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model

def cnn_model1(input_shape):
    input_img = Input(shape=input_shape)
    # 1st 2D-Convolution Layer
    conv_1 = Conv2D(16, (2, 2), padding='valid', activation='relu')(input_img)
    # 1st Max Pooling Layer
    maxpool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    # 2nd 2D-Convolution Layer
    conv_2 = Conv2D(32, (2, 2), padding='valid', activation='relu')(maxpool_1)
    # 2nd Max Pooling Layer
    maxpool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    # 3rd 2D-Convolution Layer
    conv_3 = Conv2D(64, (2, 2), padding='valid', activation='relu')(maxpool_2)
    # 3rd Max Pooling Layer
    maxpool_3 = MaxPooling2D((2, 2))(conv_3)
    # Global Average Pooling Layer
    globalavgpool = GlobalAveragePooling2D()(maxpool_3)
    # Fully-Connected Layer
    out = Dense(120, activation='softmax')(globalavgpool)
    model = Model(inputs=input_img, outputs=out)

    return model

# model summary
input_shape = (224, 224, 3)
model = cnn_model1(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

model.summary()

> Model: "model_1"
> _________________________________________________________________
> Layer (type)                 Output Shape              Param #   
> =================================================================
> input_1 (InputLayer)         (None, 224, 224, 3)       0         
> _________________________________________________________________
> conv2d_1 (Conv2D)            (None, 223, 223, 16)      208       
> _________________________________________________________________
> max_pooling2d_1 (MaxPooling2 (None, 111, 111, 16)      0         
> _________________________________________________________________
> conv2d_2 (Conv2D)            (None, 110, 110, 32)      2080      
> _________________________________________________________________
> max_pooling2d_2 (MaxPooling2 (None, 55, 55, 32)        0         
> _________________________________________________________________
> conv2d_3 (Conv2D)            (None, 54, 54, 64)        8256      
> _________________________________________________________________
> max_pooling2d_3 (MaxPooling2 (None, 27, 27, 64)        0         
> _________________________________________________________________
> global_average_pooling2d_1 ( (None, 64)                0         
> _________________________________________________________________
> dense_1 (Dense)              (None, 120)               7800      
> =================================================================
> Total params: 18,344
> Trainable params: 18,344
> Non-trainable params: 0
> _________________________________________________________________
```

Fit the first model.

```
# first model training
from keras.callbacks import ModelCheckpoint  

epochs = 10
checkpointer = ModelCheckpoint(filepath='/content/models/weights_scratch_model1.hdf5', verbose=1, save_best_only=True)
model.fit(train_tensors, train_labels,
          validation_data=(valid_tensors, valid_labels),
          epochs=epochs, batch_size=32, callbacks=[checkpointer], verbose=1)

> Train on 16567 samples, validate on 1903 samples
> Epoch 1/10
> 16567/16567 [==============================] - 76s 5ms/step - loss: 4.7849 - acc: 0.0123 - val_loss: 4.7796
> - val_acc: 0.0168
> ...
> Epoch 10/10
> 16567/16567 [==============================] - 76s 5ms/step - loss: 4.5336 - acc: 0.0424 - val_loss: 4.5774
> - val_acc: 0.0373
> 
> Epoch 00010: val_loss improved from 4.60590 to 4.57743, saving model to /content/models/weights_scratch_model1.hdf5
> <keras.callbacks.History at 0x7f6a70a77ef0>
```

Define a helper function to calculate the prediction accuracy in training, validation, and test sets.

```
# helper function to print model performance
def setAccuracy(set_tensors, set_labels):
    # breed prediction index for test set images
    breed_preds = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in set_tensors]
    
    set_accuracy = np.sum(np.array(breed_preds) == np.argmax(set_labels, axis=1)) / len(breed_preds) * 100
    print('accuracy: %.4f%%' % set_accuracy)
```

Compute the set accuracies for the first model.

```
# first model performance
model.load_weights('/content/models/weights_scratch_model1.hdf5')

print('Training Set:')
setAccuracy(train_tensors, train_labels)

print('\nValidation Set:')
setAccuracy(valid_tensors, valid_labels)

print('\nTesting Set:')
setAccuracy(test_tensors, test_labels)

> Training Set:
> accuracy: 4.7082%
> 
> Validation Set:
> accuracy: 3.7310%
> 
> Testing Set:
> accuracy: 4.1232%
```

The first model achieves a testing accuracy of only 4.1%. This is very low (but still better than random guessing at 0.8%). Judging by the accuracy of the training, validation, and testing sets, model overfit is low. The accuracy values are relatively close to one another. Still, this self-built model would not be effective at helping people determine dog breed. 

I'll adjust some of the model hyperparameters in the second model.

#### 2nd Model: Add an Inception layer to the CNN.

**Model 2 Details:**

* The second model has 3 convolutional layers with different **filter sizes** that decrease with network depth (5 x 5, 3 x 3, 1 x 1).
* The 2nd convolution layer has a `stride` (**Stride**) value (2 x 2) that deviates from the default. 
* The **activation function** remains `relu`. 
* The **padding** type used in each convolution layer is changed to`same` (Same), meaning border information is preserved and image size will not change.
* The 1st and 2nd convolution layers have a **max-pooling layer** with a low filter size (2 x 2) following them to decrease the number of parameters (and dimensionality). The 3rd convolution layer has a **flatten layer** following it to collapse the spatial dimensions of the input into the channel dimension.
* The **number of layers** in the convolution layers is held constant (4).
* After the 2nd convolution layer but before the 2nd pooling layer an **inception layer** is placed. An inception layer concatentes multiple convolution layers (with different filter sizes) and pooling layers operating in parallel within the same layer. Inception layers allow a neural network to learn better weights and select more useful features. This inception layer is comprised of 5 conception layers and 1 pooling layer, all with different hyperparameter settings.
* At the end of the network, the `softmax` (Softmax) **activation function** is applied to normalize input into a probability distribution of class probabilities.
* The **optimizer** remains `adam` (Adam). 
* The **loss** remains `categorical_crossentropy` (Categorical Cross-Entropy).
* The performance **metric** remains `accuracy` (Accuracy).
* The fit will also use an `input_shape` (**Image Size**) of (224, 224, 3), 10 `epochs` (**Epochs**) and a `batch_size` (**Batch Size**) of 32.

Construct the second model architecture.

```
# second model architecture (add inception layer)
def cnn_model2(input_shape):
    input_img = Input(shape=input_shape)
    # 1st 2D-Convolution Layer
    conv_1 = Conv2D(4, (5, 5), strides=(1, 1), padding='same', activation='relu')(input_img)
    # 1st Max-Pooling Layer
    maxpool_1 = MaxPooling2D((2, 2))(conv_1)
    # 2nd 2D-Convolution Layer
    conv_2 = Conv2D(4, (3, 3), strides=(2, 2), padding='same', activation='relu')(maxpool_1)
		
    # Inception Layer
    incept_1 = Conv2D(8, (1, 1), padding='same', activation='relu')(conv_2)
    incept_1 = Conv2D(8, (3, 3), padding='same', activation='relu')(incept_1)
    incept_2 = Conv2D(8, (1, 1), padding='same', activation='relu')(conv_2)
    incept_2 = Conv2D(8, (5, 5), padding='same', activation='relu')(incept_2)
    incept_3 = MaxPooling2D((3, 3), padding='same')(conv_2)
    incept_3 = Conv2D(8, (1, 1), padding='same', activation='relu')(incept_3)
    incept_out = concatenate([incept_1, incept_2, incept_3], axis=3)
		
    # 2nd Max-Pooling Layer
    maxpool_2 = MaxPooling2D((2, 2))(incept_out)
    # 3rd 2D-Convolution Layer
    conv_3 = Conv2D(4, (1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool_2)
    # 1st Flatten Layer
    flatten_1 = Flatten()(conv_3)
    # Fully-Connected Layer
    out = Dense(120, activation='softmax')(flatten_1)
    model = Model(inputs=input_img, outputs=out)

    return model

# model summary
input_shape = (224, 224, 3)
model = cnn_model2(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

> Model: "model_2"
> __________________________________________________________________________________________________
> Layer (type)                    Output Shape         Param #     Connected to                     
> ==================================================================================================
> input_2 (InputLayer)            (None, 224, 224, 3)  0                                            
> __________________________________________________________________________________________________
> conv2d_4 (Conv2D)               (None, 224, 224, 4)  304         input_2[0][0]                    
> __________________________________________________________________________________________________
> max_pooling2d_4 (MaxPooling2D)  (None, 112, 112, 4)  0           conv2d_4[0][0]                   
> __________________________________________________________________________________________________
> conv2d_5 (Conv2D)               (None, 56, 56, 4)    148         max_pooling2d_4[0][0]            
> __________________________________________________________________________________________________
> conv2d_6 (Conv2D)               (None, 56, 56, 8)    40          conv2d_5[0][0]                   
> __________________________________________________________________________________________________
> conv2d_8 (Conv2D)               (None, 56, 56, 8)    40          conv2d_5[0][0]                   
> __________________________________________________________________________________________________
> max_pooling2d_5 (MaxPooling2D)  (None, 56, 56, 4)    0           conv2d_5[0][0]                   
> __________________________________________________________________________________________________
> conv2d_7 (Conv2D)               (None, 56, 56, 8)    584         conv2d_6[0][0]                   
> __________________________________________________________________________________________________
> conv2d_9 (Conv2D)               (None, 56, 56, 8)    1608        conv2d_8[0][0]                   
> __________________________________________________________________________________________________
> conv2d_10 (Conv2D)              (None, 56, 56, 8)    40          max_pooling2d_5[0][0]            
> __________________________________________________________________________________________________
> concatenate_1 (Concatenate)     (None, 56, 56, 24)   0           conv2d_7[0][0]                   
>                                                                  conv2d_9[0][0]                   
>                                                                  conv2d_10[0][0]                  
> __________________________________________________________________________________________________
> max_pooling2d_6 (MaxPooling2D)  (None, 28, 28, 24)   0           concatenate_1[0][0]              
> __________________________________________________________________________________________________
> conv2d_11 (Conv2D)              (None, 28, 28, 4)    100         max_pooling2d_6[0][0]            
> __________________________________________________________________________________________________
> flatten_1 (Flatten)             (None, 3136)         0           conv2d_11[0][0]                  
> __________________________________________________________________________________________________
> dense_2 (Dense)                 (None, 120)          376440      flatten_1[0][0]                  
> ==================================================================================================
> Total params: 379,304
> Trainable params: 379,304
> Non-trainable params: 0
> __________________________________________________________________________________________________
```

Fit the second model.

```
# second model fit
epochs = 10
checkpointer = ModelCheckpoint(filepath='/content/models/weights_scratch_model2.hdf5', verbose=1, save_best_only=True)
model.fit(train_tensors, train_labels,
          validation_data=(valid_tensors, valid_labels),
          epochs=epochs, batch_size=32, callbacks=[checkpointer], verbose=1)

> Train on 16567 samples, validate on 1903 samples
> Epoch 1/10
> 16567/16567 [==============================] - 119s 7ms/step - loss: 4.7618 - acc: 0.0134 - val_loss: 
> 4.7033 - val_acc: 0.0179
> ...
> Epoch 10/10
> 16567/16567 [==============================] - 119s 7ms/step - loss: 0.3364 - acc: 0.9244 - val_loss: > 
> 12.8582 - val_acc: 0.0247
> 
Epoch 00010: val_loss did not improve from 4.61382
<keras.callbacks.History at 0x7f69f91c2080>
```

Compute the set accuracies for the second model.

```
# second model performance
model.load_weights('/content/models/weights_scratch_model2.hdf5')

print('Training Set:')
setAccuracy(train_tensors, train_labels)

print('\nValidation Set:')
setAccuracy(valid_tensors, valid_labels)

print('\nTesting Set:')
setAccuracy(test_tensors, test_labels)

> Training Set:
> accuracy: 8.2091%

> Validation Set:
> accuracy: 3.0478%

> Testing Set:
> accuracy: 2.9858%
```

The second model performs worse. The testing set accuracy is down from 4.1% to 3.0%. As even more bad news, there is a clear sign of overfitting. The training set accuracy is 8% while the validation set accuracy is lower than half of that, at 3%. This is an obvious indication that the model is tuned too closely to training data and fails to generalize well. The adjustments made in the second model are clearly not the way to go.

I'll tweak this model to see if improvements can be made in the third model.

#### 3rd Model: Increase the layer channels.

**Model 3 Details:**

* The third model has 3 convolutional layers where the 3rd convolutional layer has a different **filter size** (5 x 5, 5 x 5, 1 x 1).
* The 1st and 2nd convolution layers have a `stride` (**Stride**) value (2 x 2) that deviates from the default. 
* The **activation function** remains `relu`. 
* The **padding** type used in each convolution layer is changed back to `valid` (Valid). The padding in the inception layer remains set to `same` (Same).
* The 1st and 2nd convolution layers have a **max-pooling layer** with a higher (by 1 unit) filter size (3 x 3) following them to decrease the number of parameters. The 3rd convolution layer still has a **flatten layer** following it to collapse the spatial dimensions of the input into the channel dimension.
* The **number of layers** in the convolution layers is held constant at twice the value (8) of the last model (4).
* After the 2nd convolution layer but before the 2nd pooling layer there is still an **inception layer**. This inception layer has convolution sub-layers with twice the **number of filters** than the second model. All other settings are the same.
* At the end of the network, the `softmax` (Softmax) **activation function** is applied to normalize input into a probability distribution of class probabilities.
* The **optimizer** remains `adam` (Adam). 
* The **loss** remains `categorical_crossentropy` (Categorical Cross-Entropy).
* The performance **metric** remains `accuracy` (Accuracy).
* The fit will use an `input_shape` (**Image Size**) of (224, 224, 3), 10 `epochs` (**Epochs**) and a `batch_size` (**Batch Size**) of 32.

Construct the third model architecture.

```
# third model architecture (increase channels)
def cnn_model3(input_shape):
    input_img = Input(shape=input_shape)
    # 1st 2D-Convolution Layer
    conv_1 = Conv2D(8, (5, 5), strides=(2, 2), padding='valid', activation='relu')(input_img)
    # 1st Max-Pooling Layer
    maxpool_1 = MaxPooling2D((3, 3), padding='valid')(conv_1)
    # 2nd 2D-Convolution Layer
    conv_2 = Conv2D(8, (5, 5), strides=(2, 2), padding='valid', activation='relu')(maxpool_1)
		
    # Inception Layer
    incept_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(conv_2)
    incept_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(incept_1)
    incept_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(conv_2)
    incept_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(incept_2)
    incept_3 = MaxPooling2D((3, 3), padding='same')(conv_2)
    incept_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(incept_3)
    inception_out = concatenate([incept_1, incept_2, incept_3], axis=3)
		
    # 2nd Max-Pooling Layer
    maxpool_2 = MaxPooling2D((3, 3), padding='valid')(inception_out)
    # 3rd 2D-Convolution Layer
    conv_3 = Conv2D(8, (1, 1), padding='same', activation='relu')(maxpool_2)
    # 1st Flatten Layer
    flatten_1 = Flatten()(conv_3)
    # Fully-Connected Layer
    out = Dense(120, activation='softmax')(flatten_1)
    model = Model(inputs=input_img, outputs=out)
    return model

# model summary
input_shape = (224, 224, 3)
model = cnn_model3(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

> Model: "model_3"
> __________________________________________________________________________________________________
> Layer (type)                    Output Shape         Param #     Connected to                     
> ==================================================================================================
> input_3 (InputLayer)            (None, 224, 224, 3)  0                                            
> __________________________________________________________________________________________________
> conv2d_12 (Conv2D)              (None, 110, 110, 8)  608         input_3[0][0]                    
> __________________________________________________________________________________________________
> max_pooling2d_7 (MaxPooling2D)  (None, 108, 108, 8)  0           conv2d_12[0][0]                  
> __________________________________________________________________________________________________
> conv2d_13 (Conv2D)              (None, 52, 52, 8)    1608        max_pooling2d_7[0][0]            
> __________________________________________________________________________________________________
> conv2d_14 (Conv2D)              (None, 52, 52, 16)   144         conv2d_13[0][0]                  
> __________________________________________________________________________________________________
> conv2d_16 (Conv2D)              (None, 52, 52, 16)   144         conv2d_13[0][0]                  
> __________________________________________________________________________________________________
> max_pooling2d_8 (MaxPooling2D)  (None, 52, 52, 8)    0           conv2d_13[0][0]                  
> __________________________________________________________________________________________________
> conv2d_15 (Conv2D)              (None, 52, 52, 16)   2320        conv2d_14[0][0]                  
> __________________________________________________________________________________________________
> conv2d_17 (Conv2D)              (None, 52, 52, 16)   6416        conv2d_16[0][0]                  
> __________________________________________________________________________________________________
> conv2d_18 (Conv2D)              (None, 52, 52, 16)   144         max_pooling2d_8[0][0]            
> __________________________________________________________________________________________________
> concatenate_2 (Concatenate)     (None, 52, 52, 48)   0           conv2d_15[0][0]                  
>                                                                  conv2d_17[0][0]                  
>                                                                  conv2d_18[0][0]                  
> __________________________________________________________________________________________________
> max_pooling2d_9 (MaxPooling2D)  (None, 50, 50, 48)   0           concatenate_2[0][0]              
> __________________________________________________________________________________________________
> conv2d_19 (Conv2D)              (None, 50, 50, 8)    392         max_pooling2d_9[0][0]            
> __________________________________________________________________________________________________
> flatten_2 (Flatten)             (None, 20000)        0           conv2d_19[0][0]                  
> __________________________________________________________________________________________________
> dense_3 (Dense)                 (None, 120)          2400120     flatten_2[0][0]                  
> ==================================================================================================
> Total params: 2,411,896
> Trainable params: 2,411,896
> Non-trainable params: 0
> __________________________________________________________________________________________________
```

Fit the third model.

```
# third model fit
epochs = 10
checkpointer = ModelCheckpoint(filepath='/content/models/weights_scratch_model3.hdf5', verbose=1, save_best_only=True)
model.fit(train_tensors, train_labels,
          validation_data=(valid_tensors, valid_labels),
          epochs=epochs, batch_size=32, callbacks=[checkpointer], verbose=1)
					
> Train on 16567 samples, validate on 1903 samples
> Epoch 1/10
> 16567/16567 [==============================] - 61s 4ms/step - loss: 4.7861 - acc: 0.0120 - val_loss: 4.7839
>  - val_acc: 0.0116
>  ...
>  Epoch 10/10
> 16567/16567 [==============================] - 59s 4ms/step - loss: 0.2058 - acc: 0.9617 - val_loss: 
> 13.9675 - val_acc: 0.0310
>
> Epoch 00010: val_loss did not improve from 4.65657
> <keras.callbacks.History at 0x7f69b8fa72b0>
```

Compute the set accuracies for the third model.

```
# third model performance
model.load_weights('/content/models/weights_scratch_model3.hdf5')

print('Training Set:')
setAccuracy(train_tensors, train_labels)

print('\nValidation Set:')
setAccuracy(valid_tensors, valid_labels)

print('\nTesting Set:')
setAccuracy(test_tensors, test_labels)

> Training Set:
> accuracy: 6.6820%
> 
> Validation Set:
> accuracy: 3.1529%
> 
> Testing Set:
> accuracy: 1.9431%
```

Unfortunate news. The test accuracy has decreased yet again. The value fell from 4.1% to 3.0%, and now to 1.9%. An inception layer and 3 alternating convolution-pooling layers does not seem like the way to approach this problem. On top of this, overfitting is still present in the model. 

I'll change architectures completely in the fourth self-built model.

#### 4th Model: Change to a CNN with batch normalization.

**Model 4 Details:**

* The fourth model has 4 convolutional layers with equal **filter sizes** (3 x 3).
* The `stride` (**Stride**) values are set to the default (1 x 1). 
* The **activation function** remains `relu`. 
* The **padding** type used in each convolution layer is changed back to `same` (Same). 
* The inception layer is removed.
* The 1st, 2nd, and 3rd convolution layers have a **max-pooling layer** with a higher (by 1 unit) filter size (4 x 4) and large stride (4 x 4) following them to decrease the number of parameters and input size. The 4th convolution layer has a **flatten layer** following it to collapse the spatial dimensions of the input into the channel dimension.
* **Batch normalization** layers are applied to each convolution layer (before the activation functions) in order to normalize the input by adjusting and scaling the activations. This stabilizes the learning process and reduces the number of training epochs required to achieve optimal results.
* **Dropout layers** are used after the pooling (and flatten) layers to prevent the model from forming a dependency on particular features. This will protect against overfitting.
* The **number of layers** in the convolution layers doubles with network depth (16, 32, 64, 128).
* At the end of the network, the `softmax` (Softmax) **activation function** is applied to normalize input into a probability distribution of class probabilities.
* The **optimizer** remains `adam` (Adam). 
* The **loss** remains `categorical_crossentropy` (Categorical Cross-Entropy).
* The performance **metric** remains `accuracy` (Accuracy).
* The fit will use an `input_shape` (**Image Size**) of (224, 224, 3), 10 `epochs` (**Epochs**) and a `batch_size` (**Batch Size**) of 32.

Construct the fourth model architecture.

```
# fourth model architecture (add batch normalization)
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.normalization import BatchNormalization
def cnn_model4(input_shape):
    input_img = Input(shape=input_shape)
		
    # 1st 2D-Convolution Layer
    conv_1 = Conv2D(16, (3, 3), padding='same', use_bias=False)(input_img)		
    # 1st Batch Normalization Layer
    batch_1 = BatchNormalization(axis=3, scale=False)(conv_1)		
    # 1st Activation Layer
    activation_1 = Activation('relu')(batch_1)
    # 1st Max-Pooling Layer
    maxpool_1 = MaxPooling2D((4, 4), strides=(4, 4), padding='same')(activation_1)
    # 1st Dropout Layer
    dropout_1 = Dropout(0.2)(maxpool_1)
		
    # 2nd 2D-Convolution Layer
    conv_2 = Conv2D(32, (3, 3), padding='same', use_bias=False)(dropout_1)
    # 2nd Batch Normalization Layer
    batch_2 = BatchNormalization(axis=3, scale=False)(conv_2)		
    # 2nd Activation Layer
    activation_2 = Activation('relu')(batch_2)
    # 2nd Max-Pooling Layer
    maxpool_2 = MaxPooling2D((4, 4), strides=(4, 4), padding='same')(activation_2)
    # 2nd Dropout Layer
    dropout_2 = Dropout(0.2)(maxpool_2)
		
    # 3rd 2D-Convolution Layer
    conv_3 = Conv2D(64, (3, 3), padding='same', use_bias=False)(dropout_2)
    # 3rd Batch Normalization Layer
    batch_3 = BatchNormalization(axis=3, scale=False)(conv_3)
    # 3rd Activation Layer
    activation_3 = Activation('relu')(batch_3)
    # 3rd Max-Pooling Layer
    maxpool_3 = MaxPooling2D((4, 4), strides=(4, 4), padding='same')(activation_3)
    # 3rd Dropout Layer
    dropout_3 = Dropout(0.2)(maxpool_3)
		
    # 4th Convolution Layer
    conv_4 = Conv2D(128, (3, 3), padding='same', use_bias=False)(dropout_3)
    # 4th Batch Normalization Layer
    batch_4 = BatchNormalization(axis=3, scale=False)(conv_4)
    # 4th Activation Layer
    activation_4 = Activation('relu')(batch_4)		
		
    # 1st Flatten Layer
    flatten_1 = Flatten()(activation_4)
    # 4th Dropout Layer
    dropout_4 = Dropout(0.2)(flatten_1)
    # 1st Dense Layer
    dense_1 = Dense(512, activation='relu')(dropout_4)
    # Fully-Connected Layer
    out = Dense(120, activation='softmax')(dense_1)
    model = Model(inputs=input_img, outputs=out)
    return model

# model summary
input_shape = (224, 224, 3)
model = cnn_model4(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

> Model: "model_4"
> _________________________________________________________________
> Layer (type)                 Output Shape              Param #   
> =================================================================
> input_4 (InputLayer)         (None, 224, 224, 3)       0         
> _________________________________________________________________
> conv2d_6 (Conv2D)            (None, 224, 224, 16)      432       
> _________________________________________________________________
> batch_normalization_4 (Batch (None, 224, 224, 16)      48        
> _________________________________________________________________
> activation_3 (Activation)    (None, 224, 224, 16)      0         
> _________________________________________________________________
> max_pooling2d_3 (MaxPooling2 (None, 56, 56, 16)        0         
> _________________________________________________________________
> dropout_3 (Dropout)          (None, 56, 56, 16)        0         
> _________________________________________________________________
> conv2d_7 (Conv2D)            (None, 56, 56, 32)        4608      
> _________________________________________________________________
> batch_normalization_5 (Batch (None, 56, 56, 32)        96        
> _________________________________________________________________
> activation_4 (Activation)    (None, 56, 56, 32)        0         
> _________________________________________________________________
> max_pooling2d_4 (MaxPooling2 (None, 14, 14, 32)        0         
> _________________________________________________________________
> dropout_4 (Dropout)          (None, 14, 14, 32)        0         
> _________________________________________________________________
> conv2d_8 (Conv2D)            (None, 14, 14, 64)        18432     
> _________________________________________________________________
> batch_normalization_6 (Batch (None, 14, 14, 64)        192       
> _________________________________________________________________
> activation_5 (Activation)    (None, 14, 14, 64)        0         
> _________________________________________________________________
> max_pooling2d_5 (MaxPooling2 (None, 4, 4, 64)          0         
> _________________________________________________________________
> dropout_5 (Dropout)          (None, 4, 4, 64)          0         
> _________________________________________________________________
> conv2d_9 (Conv2D)            (None, 4, 4, 128)         73728     
> _________________________________________________________________
> batch_normalization_7 (Batch (None, 4, 4, 128)         384       
> _________________________________________________________________
> activation_6 (Activation)    (None, 4, 4, 128)         0         
> _________________________________________________________________
> flatten_1 (Flatten)          (None, 2048)              0         
> _________________________________________________________________
> dropout_6 (Dropout)          (None, 2048)              0         
> _________________________________________________________________
> dense_1 (Dense)              (None, 512)               1049088   
> _________________________________________________________________
> dense_2 (Dense)              (None, 120)               61560     
> =================================================================
> Total params: 1,208,568
> Trainable params: 1,208,088
> Non-trainable params: 480
> _________________________________________________________________
```

Fit the fourth model.

```
# fourth model fit
epochs = 10
checkpointer = ModelCheckpoint(filepath='/content/models/weights_scratch_model4.hdf5', verbose=1, save_best_only=True)
model.fit(train_tensors, train_labels,
          validation_data=(valid_tensors, valid_labels),
          epochs=epochs, batch_size=32, callbacks=[checkpointer], verbose=1)
					
> Train on 16567 samples, validate on 1903 samples
> Epoch 1/10
> 16567/16567 [==============================] - 24s 1ms/step - loss: 4.6983 - acc: 0.0202 - val_loss: 4.5241
> - val_acc: 0.0305
> ...
> Epoch 10/10
> 16567/16567 [==============================] - 16s 984us/step - loss: 3.7119 - acc: 0.1156 - val_loss: > 
> 3.8958 - val_acc: 0.0935

Epoch 00010: val_loss improved from 4.14022 to 3.89578, saving model to /content/models/weights_scratch_model4.hdf5
<keras.callbacks.History at 0x7f80e0134518>
```

Compute the set accuracies for the fourth model.

```
# fourth model performance
model.load_weights('/content/models/weights_scratch_model4.hdf5')

print('Training Set:')
setAccuracy(train_tensors, train_labels)

print('\nValidation Set:')
setAccuracy(valid_tensors, valid_labels)

print('\nTesting Set:')
setAccuracy(test_tensors, test_labels)

> Training Set:
> accuracy: 11.8730%

> Validation Set:
> accuracy: 9.3537%

> Testing Set:
> accuracy: 10.1896%
```

Finally some good news! The testing set accuracy of the fourth model has increased to 10.2%. This is more than double the best performance so far. Additionally, the threat of overfitting has drastically reduced. The accuracy values between training, validation, and testing sets are all fairly close.

I'll use a pre-trained network in the last model as an example of the drastic performance gains possible with transfer knowledge.

#### 5th Model: Transfer learning with Inception-V3.

Import and save the fifth model pre-trained layers.

```
# fifth model pre-trained layers
from keras.applications import inception_v3

inception_transfer = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
train_inception = inception_transfer.predict(train_tensors, batch_size=32, verbose=1)
valid_inception = inception_transfer.predict(valid_tensors, batch_size=32, verbose=1)
test_inception = inception_transfer.predict(test_tensors, batch_size=32, verbose=1)

# fifth model weights save
np.save('/content/models/train_inception.npy', train_inception) 
np.save('/content/models/valid_inception.npy', valid_inception) 
np.save('/content/models/test_inception.npy', test_inception)
```

**Model 5 Details:**
* The input tensors are sent through the Inception-V3 pre-trained model. Inception-V3 is a CNN trained on more than a million images from the [ImageNet](http://www.image-net.org/) database. The network is comprised of 48 layers and can classify images into 1,000 classes of objects. The original motivation for this network was to design a model to avoid representational bottlenecks, where reducing the dimensions too much causes loss of information. 
* At the end of the pre-trained network,  a **dropout layer** is used to guard against overfitting and the `softmax` (Softmax) **activation function** is applied to normalize input into a probability distribution of class probabilities.
* The **optimizer** remains `adam` (Adam). 
* The **loss** remains `categorical_crossentropy` (Categorical Cross-Entropy).
* The performance **metric** remains `accuracy` (Accuracy).
* The fit will use an `input_shape` (**Image Size**) of (299, 299, 3), 10 `epochs` (**Epochs**) and a `batch_size` (**Batch Size**) of 32.

Construct the fifth model architecture and apply the fit.

```
# fifth model architecture (transfer learning)

# pre-trained layers
train_inception = np.load('/content/models/train_inception.npy') 
valid_inception = np.load('/content/models/valid_inception.npy') 
test_inception = np.load('/content/models/test_inception.npy') 

model = Sequential()  
# 1st Dense Layer
model.add(Dense(256, activation='relu'))  
# 1st dropout layer
model.add(Dropout(0.5))  
# Fully-Connected Layer
model.add(Dense(120, activation='softmax'))  

# fifth model fit
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])   
history = model.fit(train_inception, train_labels,  
          epochs=10,  
          batch_size=32,  
          validation_data=(valid_inception, valid_labels))     
model.save_weights('/content/models/weights_transfer_inceptionv3.hdf5')  

> Train on 16567 samples, validate on 1903 samples
> Epoch 1/10
> 16567/16567 [==============================] - 8s 482us/step - loss: 1.7287 - acc: 0.5815 - val_loss: 0.7815
>  - val_acc: 0.7714
>  ...
>  Epoch 10/10
> 16567/16567 [==============================] - 3s 178us/step - loss: 0.6476 - acc: 0.7981 - val_loss: 0.7566
>  - val_acc: 0.7856
> 1903/1903 [==============================] - 0s 83us/step
```

Compute the set accuracies for the fifth model.

```
# fifth model performance
model.load_weights('/content/models/weights_transfer_inceptionv3.hdf5')

print('Training Set:')
setAccuracy(train_inception, train_labels)

print('\nValidation Set:')
setAccuracy(valid_inception, valid_labels)

print('\nTesting Set:')
setAccuracy(test_inception, test_labels)

> Training Set:
> accuracy: 89.3825%
> 
> Validation Set:
> accuracy: 78.5602%
> 
> Testing Set:
> accuracy: 78.1991%
```

Transfer learning has increased the testing set accuracy from 10.2% to an astounding 78.2%. This means nearly 4 out of 5 dog breeds are correctly identified. The training set accuracy is 89.4%, indicating a slight overfit. The closeness in performance between validation and testing sets, however, provides a reason to feel good about the ability of this model to generalize to new data. 

As a last step, I'll plot the accuracy and loss by epoch for the Inception-V3 model.

```
# fifth model plots
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 12))
# epoch accuracy
plt.subplot(211)  
plt.plot(history.history['acc'], color='cornflowerblue', linewidth=3)  
plt.plot(history.history['val_acc'], color='goldenrod', linestyle='-.', linewidth=3)  
plt.title('InceptionV3 Epoch Accuracy', fontsize=18)  
plt.ylabel('Accuracy', fontsize=14)  
plt.xlabel('Epoch', fontsize=14)  
plt.legend(['Training', 'Validation'], loc='lower right',  prop={'size': 14})  
  
# epoch loss   
plt.subplot(212)  
plt.plot(history.history['loss'], color='cornflowerblue', linewidth=3)  
plt.plot(history.history['val_loss'], color='goldenrod', linestyle='-.', linewidth=3)  
plt.title('InceptionV3 Epoch Loss', fontsize=18)  
plt.ylabel('Loss', fontsize=14)  
plt.xlabel('Epoch', fontsize=14)  
plt.legend(['Training', 'Validation'], loc='upper right',  prop={'size': 14})  
plt.show()
```

![Inception V3](https://github.com/monstott/Blogs/raw/master/Blog5/inceptionv3.png)

This visualizaton shows that the training set accuracy begins to surpass the validation set at epoch 8. The set split for the loss function occurs earlier, at epoch 6.

### Final thoughts.

Self-built convolutional neural network architectures are complex creatures. This project has shown just how difficult it can be to put one together. There are an incredible number of architecture and hyperparameter combinations possible. Knowledge of the effects of every change is critical. It can become frustrating to see results decline after hours of computationally expensive work in new models. As an example, doubling the number of filters in the inception layer from model 2 to model 3 was wasted effort. The testing set accuracy actually decreased. On the other hand, inputting dimensionality reduction layers, like dropout and batch normalization, in model 4 helped jump the self-built model accuracy to its highest point. 

One of the most important takeaways from this project is how incredible the performance boost can be when incorporating professionally-sourced networks. The pre-trained Inception-V3 model blew the other four models out of the water. Its ability to properly identify every 4 out of 5 dog breeds is a viable solution.  Some improvments that could be made in the modeling process include ensuring equal class sizes, increasing the number of training epochs, and adjusting batch size and regularization. 

As a result of this effort, the input of interest in convolutional neural networks has worked its way through hidden layers of confusion and mysery and output as enhanced understanding of their form and function.
