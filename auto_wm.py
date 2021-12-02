# -*- coding: utf-8 -*-
# AutoEncoder.ipynb
#%%
import cv2
import glob
import os, sys
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from numpy.random import default_rng

# Let OpenCV multithread
nslots = int(os.environ.get('NSLOTS',1))
cv2.setNumThreads(nslots)

imdir = './stargan_data/white_men/'

width = 256
height = 256
dim = (width, height)

def img_prep(file_path):
  fp = imdir+file_path
  img = cv2.imread(fp)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return resized

file_list = os.listdir(imdir) 

# Pre-allocate the train_input storage, use float32 to save space
# Something to test:  do you get the same results with float16? If
# so you'll cut RAM usage for train_input in half
train_input = np.empty((len(file_list),256,256,3), dtype=np.float32)

# Loop through the file list and insert each into train_input
for i,file in enumerate(file_list):
    img = img_prep(file)
    train_input[i,...] = img/255.0

print("training input loaded...")

# Total memory use ~9GB at this point with float32.

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

encoder_input = keras.Input(shape = (256,256,3), name="img_in")
x = keras.layers.Conv2D(256, (3,3), activation = 'relu', padding='same')(encoder_input)
x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
x = keras.layers.Conv2D(128, (3,3), activation = 'relu', padding='same')(encoder_input)
x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
x = keras.layers.Conv2D(64, (3,3), activation = 'relu', padding='same')(x)
encoder_output = keras.layers.MaxPooling2D((2,2), padding='same')(x)
encoder_output = keras.layers.Flatten()(x)
encoder_output = keras.layers.Dense(512, activation='relu')(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
decoder_input = keras.layers.Dense(576, activation='relu')(encoder_output)

x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(encoder_output)
x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2,2))(x)

decoder_output = keras.layers.Conv2D(3, (3,3), activation='relu', padding='same')(x)

autoencoder = keras.Model(encoder_input, decoder_output, name='ae')
autoencoder.summary()

autoencoder.compile(opt, loss="mse")

autoencoder.fit(train_input[0:10000], train_input[0:10000], batch_size=16, epochs=200, validation_split=0.1)
autoencoder.save("wm_autoencoder_model.model")


# test and save some random images.
rng = default_rng()
file_list = os.listdir('./stargan_data/white_men/')  
test_imgs = rng.choice(file_list,3,replace=True)

for i,test_img in enumerate(test_imgs):
    img = img_prep(test_img)
    ae_out = autoencoder.predict([img.reshape(-1,256,256,3)])[0]
    cv2.imwrite(f"wm_autoencoder_test_{i}_{test_img}", ae_out)

 

