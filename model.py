#!/usr/bin/env python3
import os
import csv
import cv2
import numpy as np
import sklearn
import copy
import random
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda
from sklearn.model_selection import train_test_split
from skimage import draw
from functools import lru_cache
import matplotlib.pyplot as plt
from helper import *

#@lru_cache(maxsize=2000)
def loadImage(image_name, doFlip):
    image = cv2.imread(image_name, cv2.IMREAD_COLOR) # Return BGR
    if doFlip:
        image = cv2.flip(image, 1)
    # Convert to YUV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image

class Sample:
    def __init__(self, image_name, driving_angle, doFlip):
        self.image_name = "data/IMG/"+image_name.split("/")[-1]
        self.driving_angle = np.clip(driving_angle, -1, 1)
        self.doFlip = doFlip
    def getImage(self):
        return loadImage(self.image_name, self.doFlip)
    def showImage(self, f = None):
        image = self.getImage()
        s = image.shape
        line_len = s[0]//2
        driving_angle = self.driving_angle
        if f is not None:
            image, driving_angle = f(image,driving_angle)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        angle = driving_angle / 360 * np.pi * 100 # Times 100 just to make it more visible
        line_y, line_x = int(line_len * np.cos(angle)), int(line_len * np.sin(angle))
        rr, cc = draw.line(s[0]-1, s[1]//2, s[0]-1-line_y, s[1]//2 + line_x)
        image[rr, cc, :] = 255

        plt.imshow(image)
        plt.title("{} degrees".format(float(driving_angle)))
        plt.show()

def loadSamples():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == "center":
                assert line[1] == "left"
                assert line[2] == "right"
                continue
            speed = float(line[6])
            driving_angle = float(line[3])
            offset_driving_angle = 0.25
            samples.append(Sample(line[0], driving_angle, False))
            samples.append(Sample(line[0], -driving_angle, True))
            samples.append(Sample(line[1], driving_angle + offset_driving_angle, False)) # Left
            samples.append(Sample(line[1], -driving_angle - offset_driving_angle, True)) # Left
            samples.append(Sample(line[2], driving_angle - offset_driving_angle, False)) # Right
            samples.append(Sample(line[2], -driving_angle + offset_driving_angle, True)) # Right
    return samples

samples = loadSamples()
random.shuffle(samples)

imageshape = (80, 280, 3)  # Trimmed and cropped image format

#showRandomImages(samples)
showDrivingAngles(samples)

#samples = duplicateSamplesToRebalanceByDrivingAngle(samples)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#showDrivingAngles(train_samples, "Training samples")
#showDrivingAngles(validation_samples, "Validation samples")

def add_random_shadow(image):
    # image is in YUV
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    shadow_mask = np.zeros(image.shape[:2])
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image[:,:,0][cond1] = image[:,:,0][cond1]*random_bright
        else:
            image[:,:,0][cond0] = image[:,:,0][cond0]*random_bright

    return image

def randomModification(image, angle):
    # Image is in YUV
    if np.random.randint(5) == 0:
        return image, angle
    # random blur
    image = cv2.GaussianBlur(image, (np.random.randint(2)*2+1,np.random.randint(2)*2+1), 0)
    # random brightness
    image[0,:,:] = np.clip(image[0,:,:] + np.random.randint(-30,30), 0, 255)
    # random shadow
    image = add_random_shadow(image)
    # add warp
    assert image.shape == (160, 320, 3)
    shift = np.random.randint(-20,20)
    shift2 = shift #np.random.randint(-20,20)
    height_shift = np.random.randint(-10,10)
    # shift is how much to move the bottom row left and right
    # shift2 is how much to move the middle left and right
    # so if shift=shift2, then we are just translating the image
    angle += (shift+shift2)*0.002
    angle = np.clip(angle, -1,1)
    h,w,ch = image.shape
    pts1 = np.float32([[20,0],[w-20,0],[20,h/2],[w-20,h/2]])
    pts2 = np.float32([[20+shift,0],[w-20+shift,0],[20+shift2,h/2+height_shift],[w-20+shift2,h/2+height_shift]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return image, angle

#showRandomImages(samples, randomModification)

def generator(samples, batch_size, is_validation):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = batch_sample.getImage()
                driving_angle = batch_sample.driving_angle
                if not is_validation:
                    image, driving_angle = randomModification(image, driving_angle)
                assert image.shape == (160, 320, 3)
                images.append(image)
                angles.append(driving_angle)

            X_train = np.array(images, dtype=np.float64)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size, is_validation = False)
validation_generator = generator(validation_samples, batch_size=batch_size, is_validation = True)

inputs = keras.layers.Input(shape=(160, 320, 3))
output = keras.layers.convolutional.Cropping2D(cropping=((55, 25), (20, 20)))(inputs)
assert output.shape[1:] == imageshape

#showLayerOutput(train_samples[0], inputs, outputs) # Very useful debug - show what the image looks like at this stage
output = Lambda(lambda x: x/127.5 - 1., input_shape=imageshape, output_shape=imageshape)(output)
#showLayerOutput(train_samples[0], inputs, outputs) # Very useful debug - show what the image looks like at this stage

output = Conv2D(filters=24, kernel_size=5, strides=2, activation='relu')(output)
output = Conv2D(filters=36, kernel_size=5, strides=2, activation='relu')(output)
output = Conv2D(filters=48, kernel_size=5, strides=2, activation='relu')(output)
output = Conv2D(filters=64, kernel_size=3, activation='relu')(output)
output = Conv2D(filters=64, kernel_size=3, activation='relu')(output)
output = Conv2D(filters=64, kernel_size=3, activation='relu')(output)
assert output.shape[1:] == (1, 26, 64) # Note that its shape in the height direction is 1
output = Flatten()(output)
output = Dense(100, activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(50, activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(20, activation='relu')(output)
output = Dropout(0.2)(output)
output = Dense(1, activation='tanh', use_bias=False)(output)
# Note we want a symmetric activation function about zero.
# So either none, tanh, step or ramp.  And no bias

model = Model(inputs=inputs, outputs=output)

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./log', histogram_freq=1, write_graph=True, write_images=True,
    embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata=None)
checkpointCallback = keras.callbacks.ModelCheckpoint(
    'model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2,
    patience=4, min_lr=0.0001)

debugCallback = DebugCallback(train_samples[:5], model)

model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
from keras.models import load_model
model = load_model('model.h5')
model.fit_generator(train_generator, epochs=200,
                    steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size,
                    callbacks=[tbCallBack, checkpointCallback, reduce_lr, debugCallback])

model.save('model.h5')


