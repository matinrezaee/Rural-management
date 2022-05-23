# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:43:57 2021

@author: MohammadSadegh KhoshghiafeRezaee
"""


import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize
import os
import numpy as np
import matplotlib.pyplot as plt

ix = 18
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TEST_PATH = 'all/test/'
test_ids = next(os.walk(TEST_PATH))[-1]
img = imread(TEST_PATH + test_ids[ix])[:,:,:IMG_CHANNELS]
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
X_test = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
X_test[0] = img



model = tf.keras.models.load_model('checkpoints/model_for_footnote.h5')
prediction = model.predict(X_test, verbose=10) 
preds_test_t = (prediction > 0.5).astype(np.uint8)

imshow(TEST_PATH + test_ids[ix])
plt.show()

imshow(np.squeeze(preds_test_t))
plt.show()