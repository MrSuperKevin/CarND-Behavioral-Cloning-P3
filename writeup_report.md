# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image2]: ./examples/center_2018_02_12_11_26_46_821.jpg "Grayscaling"
[image3]: ./examples/center_2018_03_01_23_23_18_683.jpg "Recovery Image"
[image4]: ./examples/center_2018_03_01_23_23_19_241.jpg "Recovery Image"
[image5]: ./examples/center_2018_03_01_23_23_19_935.jpg "Recovery Image"
[image6]: ./examples/center_2018_03_01_23_23_18_683.jpg "Normal Image"
[image7]: ./examples/center_2018_03_01_23_23_18_683_fliped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* record.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (The Navida training architecture)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting .

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for driving a model architecture was to use a powerful architecture and collect more training data. The data should contains the data collected from driving along the lane and the data collected from driving to the center of the road when the car is on the edge of the road(so the car can correct itself).

My first step was to use a convolution neural network model similar to the Navida architecture I thought this model might be appropriate because they have already done a lot of research on the self-driving project. And I compared the LeNet-5 and the Navida architecture turns out that the Navida architecture is better.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model add dropout into the flatten layers.

Then I trained the model and find that the car keeps getting off the road. So I did following things to improve the model:
1. Collect training data of car driving to the center of the road from edge.
2. Add flip image of the training data to balance the data so the car will not tend to drive to one side of the road.
3. Add left and right images to train the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: the curves with big degree turns, the bridge, the curves with no boundaries. To improve the driving behavior in these cases, I collected correction data on these special scenes.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The car drives swing on the road but I think adjusting the correction param of the left and right angle and adjust the dropout param will improve the model.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Normalization     	| So the input has mean zero  
| Cropping				| Crop the image to get rid of the useless data			
| Convolution 5x5	    | 24   
| RELU					| Activation			
| Convolution 5x5	    | 36
| RELU					| Activation			
| Convolution 5x5	    | 48
| RELU					| Activation
| Convolution 3x3	    | 64
| RELU					| Activation
| Convolution 3x3	    | 64
| RELU					| Activation
| Fully connected		| output 100
| Dropout				| Dropout probability 0.9
| Fully connected		| output 50
| Dropout				| Dropout probability 0.9
| Fully connected		| output 10
| Fully connected		| output 1

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road. These images show what a recovery looks like starting from right side of the road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would balance the data and fix the problem that the car tend to turn more to the left since most of the data collected from driving clockwise. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 8131 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by whether if the model is overfitting(see if the training loss is decreasing but the validation loss is increasing). I used an adam optimizer so that manually training the learning rate wasn't necessary.
