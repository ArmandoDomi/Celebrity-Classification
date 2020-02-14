# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:18:31 2020

@author: Armando
"""


# importing keras librarys
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
import os
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, History
from keras.applications.vgg19 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


num_of_test_samples = 179
batch_size =32
test_datagen=ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('./dataset/test/',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical',shuffle=False)


model = load_model('./final_implementation/model4b.50-0.11.hdf5')
scores =model.evaluate_generator(test_set,steps=32)
print(scores)

test_set.reset()

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_set, num_of_test_samples // batch_size+1)
print(len(Y_pred))
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ['Adam_Gilchrist', 'Adam_Housley', 'Adam_Irigoyen','Adam_Johnson','Adam_Lallana']
print(classification_report(test_set.classes, y_pred, target_names=target_names))
print(test_set.classes)


