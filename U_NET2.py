# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:12:46 2021

@author: MohammadSadegh KhoshghiafeRezaee
"""
import tensorflow as tf
import os
import numpy as np
import random
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
#from keras.models import load_model
#Size of image
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
#directory
TRAIN_PATH = 'all/train/'
TEST_PATH = 'all/test/'

seed = 42
np.random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[-1]

X_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/image/' + id_ + '_img.jpg')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/mask/'))[2]:
        mask_ = imread(path + '/mask/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask
    
X_test = np.zeros((len(test_ids), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
print('Done!')

#Build the model
inputs = tf.keras.layers.Input( (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS) )
s = tf.keras.layers.Lambda( lambda x: x / 255)(inputs)
#Contraction path
c1 = tf.keras.layers.Conv2D( 16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D( 16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
#
c2 = tf.keras.layers.Conv2D( 32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D( 32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
#
c3 = tf.keras.layers.Conv2D( 64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D( 64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
#
c4 = tf.keras.layers.Conv2D( 128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D( 128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
#
c5 = tf.keras.layers.Conv2D( 256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D( 256, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c5)
###
#Expansive path
u6 = tf.keras.layers.Conv2DTranspose( 128, (2,2), strides=(2,2), padding='same' )(c5)
u6 = tf.keras.layers.concatenate( [u6, c4] )
c6 = tf.keras.layers.Conv2D( 128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D( 128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c6)
#
u7 = tf.keras.layers.Conv2DTranspose( 64, (2,2), strides=(2,2), padding='same' )(c6)
u7 = tf.keras.layers.concatenate( [u7, c3] )
c7 = tf.keras.layers.Conv2D( 64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D( 64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c7)
#
u8 = tf.keras.layers.Conv2DTranspose( 32, (2,2), strides=(2,2), padding='same' )(c7)
u8 = tf.keras.layers.concatenate( [u8, c2] )
c8 = tf.keras.layers.Conv2D( 32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D( 32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c8)
#
u9 = tf.keras.layers.Conv2DTranspose( 16, (2,2), strides=(2,2), padding='same' )(c8)
u9 = tf.keras.layers.concatenate( [u9, c1], axis=3 )
c9 = tf.keras.layers.Conv2D( 16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D( 16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same' )(c9)
#
outputs = tf.keras.layers.Conv2D( 1, (1,1), activation='sigmoid' )(c9)
#
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
#Model checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_footnote.h5', verbose=1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]
result = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
model.save('checkpoints/model_for_footnote.h5')


###
#model = load_model('model_for_nuclei.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1) 
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1) 
preds_test = model.predict(X_test, verbose=1) 

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
# Create list of upsampled test masks
"""
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
"""
ix = random.randint(0, len(preds_train_t))
#print(X_train[ix])
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()
