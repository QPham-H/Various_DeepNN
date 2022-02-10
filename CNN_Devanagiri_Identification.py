"""
CNN and k-Means Clustering to Identify Devanagiri letters in a badly labeled dataset
Coded by Quoc Pham
02/2022
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

import random
import utils
import os


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, Reshape
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


# csv file with 3073 columns, column 1 is label and image is 32x32x3
file = 'devanagiri.csv'
data = pd.read_csv(file)
print("Data shape: ", data.shape)
##print(data.tail(5))

#Check for possible duplicates
data = data.drop_duplicates()
print("After dropping duplicates: ", data.shape)

# Separate label column
labels = data.pop('label')
labels = labels.to_numpy()
labels = labels[1:] # First row is a header

# Number of unique classes/labels
encoded_labels = LabelEncoder().fit_transform(labels)
unique_count = len(np.unique(encoded_labels))
print(f'Number of unique labels: {unique_count}')
# The unique count for this data set is 602! Too much compared to the 48 letters of Devanagiri
# This could mean the dataset was labeled haphazardly with mispellings and lots of inconsistency

# One-hot encoding for model training
one_hot_labels = tf.keras.utils.to_categorical(encoded_labels)
print(f'Labels one-hot shape: {one_hot_labels.shape}')


# Reconstructing the images from the row of pixels
data_rows = data.shape[0]
images = []
for i in range(1,data_rows):
    img = np.array(data.iloc[i], dtype = np.float32)
    img = img / 255
    img = img.reshape(32,32,3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    images.append(img)

images = np.array(images)
print(f'Images shape: {len(images)}')

#cv2.imshow("image",images[0])
#cv2.waitKey(0) # waits for key press to close window

# Set aside an untouched test set 
train_data, test_data, train_labels, test_labels = train_test_split(images, one_hot_labels, test_size=0.2)

BATCH = 32
EPOCHS = 5

# Build the model
def build_model(categories):
    input_layer = Input(shape=(32,32,1)) # The 32x32 size of the original input is pretty small so we will keep filters at 8
    x = Conv2D(8,(3,3),activation='relu',padding='same')(input_layer) # Padding is the same throughout so output size matches image sizes for training the autoencoder (squeeze it then expand it to see if it encodes well)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D(2, name='encoder')(x)

    x = Flatten()(encoded)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(categories,activation='softmax')(x)

    model = Model(input_layer, out)
    print(model.summary())

    return model

METRICS = [
    tf.keras.metrics.Accuracy(name='accuracy')
    ]    

#early_stopping = tf.keras.callbacks.EarlyStopping(patience = 1)

model = build_model(unique_count) # New model and new weights for each k-fold 

# Compile the model
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = METRICS
    #callbacks = early_stopping,
    )



# Train a final model
history = model.fit(
    train_data,
    train_labels,
    batch_size = BATCH,
    epochs = EPOCHS * 5
    )


# Final evaluation on held out set
scores = model.evauluate(
    test_data,
    test_labels,
    )

print(f'These are test scores: {scores}')
# We can not expect to get good test scores with the terrible labeling in this dataset

# Using k-Mean Clustering to group 48 letters of Devanagiri
