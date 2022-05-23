# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:18:42 2021

@author: MohammadSadegh KhoshghiafeRezaee
"""
import os, os.path
import shutil
import numpy as np
from tqdm import tqdm

THIS_FOLDER = os.path.dirname(os.path.basename(__file__))
print(THIS_FOLDER)

#Size of image
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

#directory
TRAIN_PATH = 'all/train/'
TEST_PATH = 'all/test/'
MASK_PATH = 'all/mask/'

seed = 42
np.random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[-1]
test_ids = next(os.walk(TEST_PATH))[-1]
mask_ids = next(os.walk(MASK_PATH))[-1]

X_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    #print(TRAIN_PATH + id_)
    os.mkdir(TRAIN_PATH + id_[:-8])
    os.mkdir(TRAIN_PATH + id_[:-8] + '/image/')
    shutil.move(path, TRAIN_PATH + id_[:-8] + '/image/')
    #print(TRAIN_PATH + id_[:-8])
    os.mkdir(TRAIN_PATH + id_[:-8] + '/mask/')
    shutil.move(MASK_PATH + id_[:-8] + '_mask_buffered.png', TRAIN_PATH + id_[:-8] + '/mask/')