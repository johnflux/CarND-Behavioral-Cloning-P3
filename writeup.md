# Behavioral Cloning

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[imageArchitecture]: ./images/architecture.png "Model Visualization"
[imageCenter]: ./images/center_driving.jpg "Center Lane Driving"
[imageRecoverLeft]: ./images/recover_from_far_left.jpg "Recovering from far left"
[imageRecoverRight]: ./images/recover_from_right.jpg "Recovering from right"
[imageLeftFlipped]: ./images/recover_from_far_left_flipped.jpg "Horizontally flipped image"
[imageHistogram]: ./images/histogram.png "Histogram of driving angles"
[imageHistogram2]: ./images/histogram_rebalanced.png "Histogram of driving angles after rebalancing"
[imageWhiteLine1]: ./images/whiteline1.png "View from camera on the left, with white line to show exaggerated driving angle"
[imageWhiteLine2]: ./images/whiteline2.png "View from camera on the right, with white line to show exaggerated driving angle"
[imageWhiteLine3]: ./images/whiteline3.png "View from camera on the right, with white line to show exaggerated driving angle"
[imageWhiteLine4]: ./images/whiteline4.png "View from camera in the center, with white line to show driving angle"
[imageLossGraph]: ./images/lossgraph.png "Loss graph"
[imageModified]: ./images/modification.png "Modified image for augmenting training data"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* helper.py some debug helper functions to avoid cluttering model.py with debug functions
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* run.mp4 for a video of the car using model.h5 on the first track
* https://youtu.be/iTBC3-e3QhA Higher quality version run.mp4
* run2.mp4 for a video of the car using model.h5 on the **second** track
* https://youtu.be/ON8VoxLioyM Higher quality version run2.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 114-127)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 111).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 184, 186 and 188).  I doubled the number of units on the second-to-last Dense layer (compared to the NVIDIA paper) because of the additonal dropout.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 210 model.py). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer combined with ReduceLROnPlateau to reduce the learning rate when there's no improvement in the validation loss for 5 epochs. (model.py line 139 and 145).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and using images from virtual cameras on the left and right hand side of the car.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolutional network followed by fully connected layers.

My first step was to use a convolution neural network model similar to the NVIDIA 'End to End Learning for Self-Driving Cars' 2016 paper.  I thought this model might be appropriate because it performs a very similar operation on images of a similar size, while using a relatively small network.  To cope with the difference in input image size, I added an additional convolutional net.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to add three small dropout layers.  These were set to 0.5, 0.5 and 0.2 dropout.

Then I binned the images into 16 bins depending on their labelled driving angle, and **initially** I duplicated the samples in each bin to balanced the bins.  However I disabled this in favour of simply increasing the amount of data.  This seem sufficient.

I also included the images from the right and left virtual images, and cropped the image to remove the image of the bonnet of the car, to prevent it from training on this.  I apply random modifications to the training images to make the system more robust.

The final step was to run the simulator to see how well the car was driving around track one. Initially there were a few spots where the vehicle fell off the track.  To improve the driving behavior in these cases, I made some new simulator recordings showing it how to correct after drifting off.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.  I modified drive.py to increase the speed from 10mph to 25mph, and it was able to remain on the track.

#### 2. Final Model Architecture

The final model architecture (model.py lines 167-189) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Details                              |
|--------------------------------------------------------------|
| Input                 | shape=(160, 320, 3)                  |
| Cropping2D            | cropping=((55, 25), (0, 0))          |
| Lambda                | Normalize between -1 and 1           |
| Conv2D                | filters=24, kernel_size=5, strides=2 |
| Activation relu       |                                      |
| Conv2D                | filters=36, kernel_size=5, strides=2 |
| Activation relu       |                                      |
| Conv2D                | filters=48, kernel_size=5, strides=2 |
| Activation relu       |                                      |
| Conv2D                | filters=64, kernel_size=3            |
| Activation relu       |                                      |
| Conv2D                | filters=64, kernel_size=3            |
| Activation relu       |                                      |
| Conv2D                | filters=64, kernel_size=3            |
| Activation relu       |   (output shape is now (?,1,31,64))  |
| Flatten               |    (output shape is now (?,1984))    |
| Dense                 | 100 units                            |
| Activation relu       |                                      |
| Dropout               | Probability 0.5                      |
| Dense                 | 50 units                             |
| Activation relu       |                                      |
| Dropout               | Probability 0.5                      |
| Dense                 | 20 units                             |
| Activation relu       |                                      |
| Dropout               | Probability 0.2                      |
| Dense                 | 1 unit                               |
| Activation tanh       | No bias                              |

Here is a visualization of the architecture

![alt text][imageArchitecture]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][imageCenter]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it made a mistake. These images show what a recovery looks like starting from the right lane:

![alt text][imageRecoverRight]

And from the far left, out of the lane:

![alt text][imageRecoverLeft]

I use the left and right virtual camera images to add to the data, adding 0.25 to the driving angle for the view on the left, and -0.25 for the view on the right. (Line 66 model.py)

To verify that the angles are correct, I drew a white line over the image representing the desired driving direction, to visually verify the data.  For example (Note that the angle is exaggerated here by 100 times to make it easier to view):
![alt text][imageWhiteLine3]
![alt text][imageWhiteLine4]
![alt text][imageWhiteLine1]
![alt text][imageWhiteLine2]

After the collection process, I had 45990 number of data points, each with three images (center, left and right camera).

To augment the data set, I also flipped images and angles thinking that this would remove the bias to always turning left. For example, here is the above image that has been flipped:

![alt text][imageLeftFlipped]

I then grouped the samples in to 16 bins by their driving angle:

![alt text][imageHistogram]

**Initially** I then duplicated the samples in the bins to oversample them such that each bin had the same number of samples, in order to reduce the bias to always drive straight.  However I found that this was redundant once I added in other data augmentation methods, so I no longer do this oversampling.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

The images are converted to YUV colorspace (in both the model.py and the drive.py), and then fed into the model. In the model, the images are preprocessed by normalizing between -1 and 1, and cropping the top and the bottom of the images to remove the sky and images of the car bonnet.  They are also cropped by 20 pixels on the left and right, to allow me to shift the image up to 20 pixels.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

For the training set only, I then added random modifications to the images in order to further augment the data.  I add a random guassian blur of 1,3 or 5 pixels, add a random horizontal shift of up to 20 pixels (adjusting the angle by 0.002 degrees per pixel shift), and a small random vertical stretch.  I also add a random shadow (line 89 to 110 in model.py).

All these modifications were done directly in the YUV colorspace.

Here's an example of an image with a random shadow placed on the right, and shifted to the right (note the duplicated border on the left).

![alt text][imageModified]

The ideal number of epochs was actually just 1 or 2 epochs.  Although the training and validation error did continue to decrease past this, no observable improvement was made to the training.

I used an adam optimizer, combined with ReduceLROnPlateau to automatically reduce the learning rate if the validation loss no longer decreases.  However because I used just 2 epochs in the final model, the ReduceLROnPlateau was effectively not used.

See: https://youtu.be/iTBC3-e3QhA   for a video of it running.

# Track 2

I also run this on track 2.  I modified drive.py to brake depending on the steering angle.  This worked really well.

It makes only a **single** small mistake on track 2 at 2:56 - it can't steer steeply enough at one point.  I'm sure this could be easily fixed by slowing down more but I simply ran out of time to test.

See run2.mp4 or watch in high resolution here: https://youtu.be/ON8VoxLioyM
