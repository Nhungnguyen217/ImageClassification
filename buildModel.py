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
