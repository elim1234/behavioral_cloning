import os
import csv
import numpy as np

i_start_from_scratch = 1
max_training_samples = 20000
num_epochs = 1
steering_offset = 0.08 # 2.0 deg
log_data_paths = ['/opt/carnd_p3/data', '/opt/carnd_p3/recovery_maneuvers']
training_data_path = '/opt/carnd_p3/data'

samples = []
for log_data_path in log_data_paths:
    with open(log_data_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader)
        for line in reader:
            samples.append(line)

np.random.shuffle(samples)

# Trim to first n samples
print(len(samples))
max_training_samples = min(len(samples), max_training_samples)
samples = samples[:max_training_samples]
print(len(samples))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import random

# Setup Keras
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.layers import Cropping2D

from scipy import ndimage

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # offset factor for center, left, or right
            offset_factor = [0, 1, -1]
            for batch_sample in batch_samples:
                # randomly select center, left, or right
                i_camera = random.randint(0, 2)
                # randomly flip image
                i_flip = random.randint(0, 1)
                try:
                    name = training_data_path + '/IMG/'+batch_sample[i_camera].split('/')[-1]
                    image = ndimage.imread(name)
                    # center_image = cv2.imread(name)                    
                    angle = float(batch_sample[3]) + offset_factor[i_camera] * steering_offset
                    if i_flip:
                        image_flipped = np.fliplr(image)
                        angle_flipped = -angle                    
                        images.append(image_flipped)
                        angles.append(angle_flipped)                    
                    else:
                        images.append(image)
                        angles.append(angle)                    
                except:
                    print('could not load' + name + '\n')

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# size of input
row, col, ch = 160, 320, 3

if i_start_from_scratch:
    print('building model from scratch. ALL PREVIOUS TRAINING DATA WILL BE LOST.')
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    # set up cropping2D layer
    model.add(Cropping2D(cropping=((65,25), (0,0))))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
else:
    print('loading existing model')
    model = load_model('model.h5')

model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples), 
                    epochs=num_epochs, verbose = 1)
model.save('model.h5')  # creates a HDF5 file 'model.h5'
