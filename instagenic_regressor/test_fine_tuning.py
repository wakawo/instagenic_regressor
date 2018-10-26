
# coding: utf-8

# In[1]:


import os
import h5py
from keras import backend as K


# In[2]:


import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from data_loader import train_generator, val_generator


# In[3]:


class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()


# In[4]:


def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


# In[7]:


# load learned mobilenet 
weights_path = './neural-image-assessment/weights/mobilenet_weights.h5'

base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base_model.output)
x = Dense(5, activation='softmax', kernel_initializer='uniform')(x)
model = Model(base_model.input, x)
model.load_weights(weights_path, by_name=True)

# freeze ~88
for layer in model.layers[:89]:
    layer.trainable = False


# In[8]:


model.summary()


# In[9]:


optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss=earth_mover_loss)

checkpoint = ModelCheckpoint('weights/mobilenet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard]

batchsize = 200
epochs = 20

model.fit_generator(train_generator(batchsize=batchsize),
                    steps_per_epoch=(250000. // batchsize),
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_generator(batchsize=batchsize),
                    validation_steps=(5000. // batchsize))

