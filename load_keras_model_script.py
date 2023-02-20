
from keras.models import load_model
import os
import numpy as np 
import cv2 
from keras.preprocessing import image
import matplotlib.pyplot as plt




def prediction(image_filename,kerasModel):
    test_image = image.load_img(image_filename, target_size = (50,50))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    prediction_probability = kerasModel.predict(test_image)

    #image_file_name = image_filename

    # return {'prediction_probability': prediction_probability, 'image_file_name':image_file_name}
    return prediction_probability



