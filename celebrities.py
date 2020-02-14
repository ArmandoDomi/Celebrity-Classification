# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:39:48 2020

@author: Armando
"""


# importing keras librarys
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt 

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, History

from keras.optimizers import SGD



# generators
train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory('./dataset/train/',
                                                 target_size = (299, 299),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 #shuffle=False
                                                 )

test_set = test_datagen.flow_from_directory('./dataset/val/',
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            #shuffle=False
                                            )


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
print(base_model.summary())


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 5 classes
predictions = Dense(5, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

print(model.summary())

for layer in base_model.layers:
    layer.trainable = False
    

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(
        training_set,
        steps_per_epoch=32,
        epochs=50,
        validation_data=test_set,
        validation_steps=32,
        #shuffle=False
        )

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
   
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True



# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])



#callbacks
checkpointer = ModelCheckpoint(filepath='model4b.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('inception.log')


history=model.fit_generator(
        training_set,
        steps_per_epoch=32,
        epochs=50,
        validation_data=test_set,
        validation_steps=32,
        callbacks=[csv_logger, checkpointer],
        #shuffle=False
        )

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()