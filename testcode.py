# --- Importing Libraries
import os

import numpy as np
import pandas as pd
#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
# %matplotlib inline
from tensorflow.keras.utils import plot_model

# Splitting data
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

# Deep Learning
import tensorflow as tf
print('TensoFlow Version: ', tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# --- Reading Data of Class Labels
path = 'trainSet'
lab = pd.read_csv('trainSet/train.csv')

# --- Visualizing countplot of the classes
# Count PLot of the samples/observations w.r.t the classes
d = dict()
class_labels = dict()
for dirs in os.listdir(path + '/myData'):
    count = len(os.listdir(path+'/myData/'+dirs))
    d[dirs+' => '+lab[lab.ClassId == int(dirs)].values[0][1]] = count
    class_labels[int(dirs)] = lab[lab.ClassId == int(dirs)].values[0][1]

plt.figure(figsize=(20, 50))
sns.barplot(y=list(d.keys()), x = list(d.values()), palette='Set3')
plt.ylabel('Label')
plt.xlabel('Count of Samples/Observations');
plt.show()

# --- Reading Image Data
# input image dimensions
img_rows, img_cols = 32, 32
# The images are RGB.
img_channels = 3
nb_classes = len(class_labels.keys())

datagen = ImageDataGenerator()
data = datagen.flow_from_directory('trainSet/myData',
                                    target_size=(32, 32),
                                    batch_size=20041,
                                    class_mode='categorical',
                                    shuffle=True)

X, y = data.next()
# Labels are one hot encoded
print(f"Data Shape   :{X.shape}\nLabels shape :{y.shape}")

# ---Sample Images of Dataset
fig, axes = plt.subplots(10, 10, figsize=(18, 18))
for i, ax in enumerate(axes.flat):
    r = np.random.randint(X.shape[0])
    ax.imshow(X[r].astype('uint8'))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Label: '+str(np.argmax(y[r])))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
    # print("Train Shape: {}\nTest Shape : {}".format(X_train.shape, X_test.shape))

# --- Customising ResNet50 model
resnet = ResNet50(weights= None, include_top=False, input_shape= (img_rows,img_cols,img_channels))

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes, activation= 'softmax')(x)
model = Model(inputs = resnet.input, outputs = predictions)

model.summary()

# ---Visualising Model Architecture
plot_model(model, show_layer_names=True, show_shapes =True, to_file='model.png', dpi=350)
print(plot_model)