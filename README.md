# **Image Classification with Multiple Classes using CNN** 
---

**Build a Image Classification Project**

The goals / steps of this project are the following:

* In this project I build a CNN, use it to make the 10 classes Image classification
* First I will do some data preprocessing with [CIFAR 10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and make the data split step
* Then I will play around with different parameters with tiny section of dataset, to try to find the relatively best choice of parameters 
* Then I will use the training set of processed CIFAR-10 dataset to train my network and at the same time make the validation during training in order to avoid overfitting
* After training with my network, I will use the saved model to test with test set, and get the final results of my network


[//]: # (Image References)

[image1]: ./examples/CIFAR-10.jpg "CIFAR-10"
[image2]: ./examples/test.png "Test"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This project is accomplished using a [ipyng](https://github.com/lc8631058/Image_Classification_with_Multiple_Classes_using_CNN/blob/master/dlnd_image_classification.ipynb) file, in order to visualize the details.

### Data Set Summary & Exploration
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here is some simple samples from this dataset:

![alt text][image1]

I used the pandas library to calculate summary statistics of the CIFAR-10 data set:

* The size of training set is ? 
```python
n_train = len(X_train)
==> 60000
```
* The size of the validation set is ?
```python
n_validation = len(X_valid)
==> 6000
```

#### 1. Data Preprocessing.

(1) First we nomalize the RGB data from value range 0 to 255 to range 0-1, this simple step is realized by function `normalize`.

(2) I use `LabelBinarizer()` from sklearn to implement the `one_hot_encode` function, to realize one hot encode for labels. Just give a Map which includes the classes your labels have, and the `LabelBinarizer` will fit the class by itself.

(3) For data randomization, because the images in dataset are already randomized, so it's not necessary to randomize it again.

(4) The function `preprocess_and_save_data` in helper.py combined all data-preprocessing functions together and after processing it will save the data as pickle file. 

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because this can eliminate the effect of colors, but after training, I found that the RGB data have better result than the grayscale data. So finally, I decided to use RGB data. 

(1) In the functions `neural_net_image_input`, `neural_net_label_input`, `neural_net_keep_prob_input` I set 3 placeholders as the inputs, labels, and keep_probability of dropout layer.

(2) In function `conv2d_maxpool` I implement conv layer and max-pool layer.

(3) Function `flatten` will flatten all the images into one vector~

(4) Function `fully_conn` realize the final fully connected layer.

(5) And we also have `output` function to generate the final output.

#### 1. Describe final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 32x32x150 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x150				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 32x32x280 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x280 |
| Fully connected		| 1024 hidden units	|
| RELU					|												|
| Fully connected		| 128 hidden units	|
| RELU					|												|
| Fully connected		| 10 hidden units	|
| Softmax				| 10 hidden units	|
 


#### 2. Describe how to train the model.

To build the model, so I use 3 convlutional layers and 2 fully-connected layers, each convolutional part is composed of convolution, relu, max_pool, batch_normalization, dropout techniche in order. The dropout probability is 0.5, and I add dropout and batch-normalization after each layer except for the last output layer. I set batch_size to 256, cause that's the maximum batch_size that my GPU resource could afford. As for learning-rate I choose 0.001 by experience.

#### 4. Describe the approach taken for finding a solution. 

My final model results were:
* training set accuracy of 81.6%
* test set accuracy of 79.15%

### Test a Model on New Images

#### 1. Here are the test results: 

  Here are some test results: 
  
  ![alt text][image2]

#### 2. Discuss the model's predictions on these test set. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|    airplane   		|    			airplane						| 
| Automobile			| Automobile      							|
| bird    			| horse 										|
| frog					| dog											|


The model was able to correctly guess 79.15% of the test images, purly guess will get a accuracy of only 10%. 
