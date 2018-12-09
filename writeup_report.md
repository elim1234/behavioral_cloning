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

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model is an abbreviated VGG network. It has 2 3x3 filters with a depth of 32 followed by 2x2 max pooling followed by 2 3x3 filters with a depth of 64 followed by 2x2 max pooling. This is followed by dense layers with 256 nodes then 128 nodes each with 30% dropout.

The model includes RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer and cropped to exclude the area above the horizon and the area with minimal information content at the bottom.

#### 2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data
Training data was chosen to keep the vehicle driving on the road. I started with the training data provided which included clockwise laps. Then I added recovery maneuvers from the edges of the track to the track center. These were counterclockwise laps. Then I added training data at the dirt turnout where the vehicle had difficulty tracking.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to start with a standard architecture VGG. I truncated the architecture to 4 CNNs because this seemed sufficient for the relatively simple geometry of detecting and interpreting lane edges.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To combat the overfitting, I applied 50% dropout to the fully-connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded recovery maneuvers from the edges of the track. I also recorded lane following maneuvers along the tricky areas of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture
The final model architecture consisted of a convolution neural network with the following layers and layer sizes . It has 2 3x3 filters with a depth of 32 followed by 2x2 max pooling followed by 2 3x3 filters with a depth of 64 followed by 2x2 max pooling. This is followed by dense layers with 256 nodes then 128 nodes each with 30% dropout.

#### 3. Creation of the Training Set & Training Process
To capture good driving behavior, I started with the training data provided. This mostly consisted of center driving clockwise around the track.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when there is tracking error. I recorded these going counterclockwise around the track to diversify the sample set.

To augment the data sat, I randomly flipped images and angles thinking that this would eliminate any directional bias in the behavior.

I also randomly used the left, center, or right cameras and added or subtracted a 2 deg steering bias depending on camera location. This angle was selected by trial and error starting with observed steering angles while manually driving.

After the collection process, I had about 20000 data points. I then preprocessed this data by normalizing and cropping it.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal method for determining the ideal number of epochs is to execute epochs until the quality of the fit tapers or the validation fit diverges from the training fit indicating overfitting. Due to the relatively lengthy duration of each epoch I tested the result on the track after each epoch. It was found that the car could complete the course after a single epoch of training, I used an adam optimizer so that manually training the learning rate wasn't necessary.