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
[image2]: ./examples/after_pre_process.png "Grayscaling"
[image4]: ./examples/test_1.png "Traffic Sign 1"
[image5]: ./examples/test_2.png "Traffic Sign 2"
[image6]: ./examples/test_3.png "Traffic Sign 3"
[image7]: ./examples/test_4.png "Traffic Sign 4"
[image8]: ./examples/test_5.png "Traffic Sign 5"
[image9]: ./examples/dataset_visul.png "dataset samples"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This project is accomplished using a [ipyng](https://github.com/lc8631058/Image_Classification_with_Multiple_Classes_using_CNN/blob/master/dlnd_image_classification.ipynb) file, in order to visualize the details.

### 1. Data Set Summary & Exploration
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here is some simple samples from this dataset:

![alt text][image1]

I used the pandas library to calculate summary statistics of the CIFAR-10 data set:

* The size of training set is ? 
```python
n_train = len(X_train)
==> 34799
```
* The size of the validation set is ?
```python
n_validation = len(X_valid)
==> 4410
```
* The size of test set is ?
```python
n_test = len(X_test)
==> 12630
``` 

#### 2. Data Preprocessing.

(1) First we nomalize the RGB data from value range 0 to 255 to range 0-1, this simple step is realized by function `normalize`.

(2) I use `LabelBinarizer()` from sklearn to implement the `one_hot_encode` function, to realize one hot encode for labels. Just give a Map which includes the classes your labels have, and the `LabelBinarizer` will fit the class by itself.

(3) For data randomization, because the images in dataset are already randomized, so it's not necessary to randomize it again.

(4) The function `preprocess_and_save_data` in helper.py combined all data-preprocessing functions together and after processing it will save the data as pickle file. 

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because this can eliminate the effect of colors, but after training, I found that the RGB data have better result than the grayscale data. So finally, I decided to use RGB data. 
Here is an example of a traffic sign image after normalizing.

![alt text][image2]

As a last step, I normalized the image data because we can't directly feed raw images to the network, the numbers are too big, so we simply divided by 255.0, to nomalize the pixel values to 0 ~ 1.

I didn't generate additional data because I don't know how to imply these techniches. Maybe later I would update my project.

#### 2. Describe final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x200 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x200 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 32x32x108 	|
| Max pooling	      	| 2x2 stride,  outputs 16x16x108 				|
| Fully connected		| 100 hidden units	|
| RELU					|												|
| Fully connected		| 43 hidden units	|
| Softmax				| 43 hidden units	|
 


#### 3. Describe how you trained your model.

To train the model, I use the model described in this [paper](https://github.com/lc8631058/SDCND/blob/master/P2-Traffic-Sign-Classifier/Traffic%20Sign%20Recognition%20with%20Multi-Scale%20Convolutional%20Networks.pdf), so I use 2 convlutional layers and 2 fully-connected layers, each convolutional part is composed of convolution, relu, max_pool, batch_normalization, dropout techniche in order. The dropout probability is 0.5, and I add dropout and batch-normalization after each layer except for the last output layer. I set batch_size to 80, cause that's the maximum batch_size that my GPU resource could afford. The number of epochs is 3000, after training for 3000 epochs I will choose the best accuracy as final accuracy, the I will re-train the network for one more time and set it to stop if the accuracy reach to the maximum value once again. As for learning-rate I choose 0.001 by experience.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.86% 
* test set accuracy of 97.29%

If a well known architecture was chosen:
* What architecture was chosen?
At first I read the [paper](https://github.com/lc8631058/SDCND/blob/master/P2-Traffic-Sign-Classifier/Traffic%20Sign%20Recognition%20with%20Multi-Scale%20Convolutional%20Networks.pdf) and directly use their apporaches. Not all of their approaches, because I'm not familiar with some preprocessing techniques described in this paper, so as a result my accuracy is not so high as theirs.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. 

Here are five German traffic signs that I found on the web:
2,  6, 17, 19, 26

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because as you can see the brightness is very low, it's hard to recognize the sign by eyes. The others should be easier to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)     		| Speed limit (50km/h)   									| 
| End of speed limit (80km/h)     			| End of speed limit (80km/h) 										|
| No entry					| No entry											|
| Dangerous curve to the left	      		| Dangerous curve to the left					 				|
| Traffic signals			| Traffic signals      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.29%.

 Actually I choose totally 45 images as the new images download from web, and the final mean accuracy of the 45 new images is 96.97%. I test them by setting evey 5 images as a batch, some of batches get 80% accuracy and some get 100%. So the above 5 images are the first batch of 45 images, and fortunately they have been predicted 100% correct. 
 
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 43th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is Dangerous curve to the right sign (probability of 9.99915719e-01), and the image does exist Dangerous curve to the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99915719e-01        			| Dangerous curve to the right  									| 
| 4.76271380e-05     				| Children crossing    										|
| 2.77236813e-05					| End of all speed and passing limits  										|
| 5.58202373e-06	      			| Pedestrians  					 				|
| 1.41421299e-06			    | Speed limit (20km/h)         							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I have encountered some error when dealing with this part, will updated later.
