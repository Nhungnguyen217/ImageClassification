import os

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import sns
from tensorflow import keras

model = keras.models.load_model('TSC_model.h5')

# --- Reading Data of Class Labels
path = 'trainSet'
lab = pd.read_csv('trainSet/train.csv')

d = dict()
class_labels = dict()
for dirs in os.listdir(path + '/myData'):
    count = len(os.listdir(path+'/myData/'+dirs))
    d[dirs+' => '+lab[lab.ClassId == int(dirs)].values[0][1]] = count
    class_labels[int(dirs)] = lab[lab.ClassId == int(dirs)].values[0][1]

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)
print("Train Shape: {}\nTest Shape : {}".format(X_train.shape, X_test.shape))

pred = np.argmax(model.predict(X_test), axis = 1)
labels = [class_labels[i] for i in range(10)]
print(classification_report(np.argmax(y_test, axis = 1), pred, target_names = labels))

# cmat = confusion_matrix(np.argmax(y_test, axis=1), pred)
# plt.figure(figsize=(16,16))
# # sns.heatmap(cmat, annot = True, cbar = False, cmap='Paired', fmt="d", xticklabels=labels, yticklabels=labels);


fig, axes = plt.subplots(5,5, figsize=(18,18))
for i,ax in enumerate(axes.flat):
    r = np.random.randint(X_test.shape[0])
    ax.imshow(X_test[r].astype('uint8'))
    ax.grid(False)
    ax.axis('off')
    ax.set_title('Original: {} Predicted: {}'.format(np.argmax(y_test[r]), np.argmax(model.predict(X_test[r].reshape(1, 32, 32, 3)))))

plt.show()