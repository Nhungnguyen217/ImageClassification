# step1: Import the required libraries

import matplotlib.pyplot as plt
import seaborn as sns

# use of the Keras library for creating our model and training it
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

# step 2: Loading the data
labels = ['Speed limit (20km/h)',
          'Speed limit (30km/h)',
          'Speed limit (50km/h)',
          'Speed limit (60km/h)',
          'Speed limit (70km/h)',
          'Speed limit (80km/h)',
          'End of speed limit (80km/h)',
          'Speed limit (100km/h)',
          'Speed limit (120km/h)',
          'No passing',
          'No passing for vechiles over 3.5 metric tons',
          'Right-of-way at the next intersection',
          'Priority road',
          'Yield',
          'Stop',
          'No vechiles',
          'Vechiles over 3.5 metric tons prohibited',
          'No entry',
          'General caution',
          'Dangerous curve to the left',
          'Dangerous curve to the right',
          'Double curve',
          'Bumpy road',
          'Slippery road',
          'Road narrows on the right',
          'Road work',
          'Traffic signals',
          'Pedestrians',
          'Children crossing',
          'Bicycles crossing',
          'Beware of ice/snow',
          'Wild animals crossing',
          'End of all speed and passing limits',
          'Turn right ahead',
          'Turn left ahead',
          'Ahead only',
          'Go straight or right',
          'Go straight or left',
          'Keep right',
          'Keep left',
          'Roundabout mandatory',
          'End of no passing',
          'End of no passing by vechiles over 3.5 metric tons',
          ]
img_size = 32

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)