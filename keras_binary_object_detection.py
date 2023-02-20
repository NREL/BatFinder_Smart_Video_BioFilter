from __future__ import print_function
import matplotlib.pyplot as plt 
import cv2
import os
import numpy as np 
import csv
from itertools import zip_longest
import numpy as np 

# from keras.datasets import bat_images
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(1671)


import tkinter as tk
from tkinter import filedialog

from tqdm import tqdm

def get_directory():
    root = tk.Tk()
    root_path = filedialog.askdirectory()
    root.withdraw()
    return root_path



############################################################
## This is the location of the training and validation data sets\


print('Please select the folder where the training data is located')
train_folder = get_directory()
print(train_folder)

print('Please select the folder where the validation data is located')
val_folder = get_directory()
print(val_folder)

print('Please select the folder where the model will be stored')
save_model = get_directory()
print(save_model)
##################################################################
os.chdir(save_model)

###########################################################################################################################################
## Here we are simply naming the model that we are producing.
model_title = 'BatFinder_Smart_Video_BioFilter'
############################################################################################################################################

#### Going to do some transformation to get more images
image_gen = ImageDataGenerator(#rotation_range = 30, # rotate the image 30 degrees
                               width_shift_range = 0.1, # shift the pic width by a max of 10%
                               height_shift_range = 0.1, # Shift the pic height by a max of 10%
                               rescale = 1/255, # Rescale the image by normailing it
                               shear_range = 0.2, # Shear means cutting away part of the iamge (max 20%)
                               zoom_range = 0.2, # Zoom in by 20% max
                               horizontal_flip = True,  # allo horizontal fliping
                               fill_mode = 'nearest' # fill in missing pixels with the nearest filled value
                               )

# plt.imshow(image_gen.random_transform(bat))
# plt.show()

image_gen.flow_from_directory(train_folder)
image_gen.flow_from_directory(val_folder)

image_shape = (50, 50, 3)
##################################################################################################
# Here we are organizing the model that will be used to run the object detection
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (50, 50, 3), activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64, kernel_size =(3,3), input_shape = (50, 50, 3), activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (50, 50, 3), activation = 'relu',))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
### Remember to set your Dense layer to the number of categories
model.add(Dense(1))
model.add(Activation('sigmoid'))




#########################################################################################################################################


model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.summary()

batch_size = 64

train_image_gen = image_gen.flow_from_directory(train_folder,
                                                target_size = image_shape[:2],
                                                batch_size = batch_size,
                                                class_mode = 'binary')

val_image_gen = image_gen.flow_from_directory(val_folder,
                                                target_size = image_shape[:2],
                                                batch_size = batch_size,
                                                class_mode = 'binary')

train_image_gen.class_indices
print(train_image_gen.class_indices)


####################################################################################
## Here we are running the model and storying the values of the accuracy in the results
import warnings
warnings.filterwarnings('ignore')

results = model.fit_generator(train_image_gen,
                              epochs = 100,
                              steps_per_epoch = 75,
                              validation_data = val_image_gen,
                              validation_steps = 120)

## Saving the model
model.save(model_title + '.h5')


##################################################

print(results.history.keys())
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

accuracy = results.history['accuracy']
val_accuracy = results.history['val_accuracy']

loss = results.history['loss']
val_loss = results.history['val_loss']

print('plot stuff done')

d = [accuracy, val_accuracy, loss, val_loss]
export_data = zip_longest(*d, fillvalue = '')
with open(model_title + '_results.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(('accuracy', 'val_accuracy', 'loss', 'val_loss'))
      wr.writerows(export_data)
myfile.close()