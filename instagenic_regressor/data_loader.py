
# coding: utf-8

# In[3]:


import numpy as np
import os
import glob
import pickle

from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf


# In[13]:


base_image_path = '../../annotation_tools/images/omelette_rice_500/images/'

IMAGE_SIZE = 224

with open('./pickle/file_list.pickle', 'rb') as f:
    files = pickle.load(f)
with open('./pickle/score_list.pickle', 'rb') as f:
    train_scores = pickle.load(f)
    
train_image_paths = [base_image_path + name for name in [files]]
train_image_paths = np.array(train_image_paths).reshape(-1)
train_scores = np.array(train_scores, dtype='float32').reshape(-1)

val_image_paths = train_image_paths[-50:]
val_scores = train_scores[-50:]
train_image_paths = train_image_paths[:-50]
train_scores = train_scores[:-50]

print('Train set size : ', train_image_paths.shape, train_scores.shape)
print('Val set size : ', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready !')


# In[ ]:


def parse_data(filename, scores):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores


# In[ ]:


def train_generator(batchsize, shuffle=True):
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset().from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)


# In[ ]:


def val_generator(batchsize):
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset().from_tensor_slices((val_image_paths, val_scores))
        val_dataset = val_dataset.map(parse_data, num_parallel_calls=2)

        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)

