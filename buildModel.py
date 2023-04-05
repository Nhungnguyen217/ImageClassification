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
path = "C:\\Users\\HP\\PycharmProjects\\Thigiacmaytinh\\ImageClassification\\trainSet\\myData"

labels = ['Speed limit (20km/h)',
          'No passing',
          'Right-of-way at the next intersection',
          'Yield',
          'Stop',
          'Traffic signals',
          'Children crossing',
          'Ahead only',
          'Go straight or left',
          'Keep right',
          ]
img_size = 32


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
