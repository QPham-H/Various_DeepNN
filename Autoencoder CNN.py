import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import random
import utils
import os


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, Reshape, UpSampling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


# csv file with 3073 columns, column 1 is label and image is 32x32x3
file = 'devanagiri.csv'
data = pd.read_csv(file)
#print("Data shape: ", data.shape)
##print(data.tail(5))

#Check for possible duplicates
data = data.drop_duplicates()
#print("After dropping duplicates: ", data.shape)

# Separate label column
labels = data.pop('label')
labels = labels.to_numpy()
labels = labels[1:] # First row is a header

# Use Labelencoder to convert to numerical encoding to run the model on (not for autoencoder function)
lencoder = LabelEncoder()
labels = lencoder.fit_transform(labels)
#print(f'Labels shape: {labels.shape}')

data_rows = data.shape[0]
images = []
for i in range(1,data_rows):
    img = np.array(data.iloc[i], dtype = np.float32)
    img = img / 255
    img = img.reshape(32,32,3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #print(img.shape)
    images.append(img)

images = np.array(images)
#images = images.reshape(-1,32,32,1)

#print(f'Images shape: {len(images)}')

##cv2.imshow("image",images[0])
##cv2.waitKey(0) # waits for key press to close window

# Set aside an untouched test set 
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)

print(f'Train data shape: {train_data.shape}')
print(f'Test data shape: {test_data.shape}')

BATCH = 32
EPOCHS = 1

# Build the model
def build_model():
    input_layer = Input(shape=(32,32,1)) # The 32x32 size of the original input is pretty small so we will keep filters at 8
    x = Conv2D(8,(3,3),activation='relu',padding='same')(input_layer) # Padding is the same throughout so output size matches image sizes for training the autoencoder (squeeze it then expand it to see if it encodes well)
    x = MaxPooling2D(2)(x)
    x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
    encoded = MaxPooling2D(2, name='encoder')(x)

    x = Conv2DTranspose(8,(3,3),activation='relu',padding='same')(encoded)
    x = UpSampling2D(2)(x)
    x = Conv2DTranspose(1,(3,3),activation='sigmoid',padding='same')(x)
    out = UpSampling2D(2)(x)

    model = Model(input_layer, out)
    print(model.summary())

    return model
"""
def build_model():
    input_layer = Input(shape=(32,32,1)) # The 32x32 size of the original input is pretty small so we will keep filters at 8
    x = Conv2D(8,(3,3),activation='relu',padding='same')(input_layer) # Padding is the same throughout so output size matches image sizes for training the autoencoder (squeeze it then expand it to see if it encodes well)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2),padding='same')(x)
    x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
    encoded = MaxPooling2D((2,2),padding='same',name='encoder')(x)

    x = Conv2D(8,(3,3),activation='relu',padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(8,(3,3),activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    out = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

    model = Model(input_layer, out)
    print(model.summary())

    return model
"""

autoencoder = build_model()

# Compile the model
autoencoder.compile(
    loss = 'mse',
    optimizer = 'adam'
    )

# Train model
history = autoencoder.fit(
    train_data,
    train_data,
    batch_size = BATCH,
    epochs = EPOCHS
    )


# Final evaluation on held out set
scores = autoencoder.evaluate(
    test_data,
    test_data,
    )

#print(f'These are test scores: {scores}')


# Create encoder for KNN querying
encoder = Model(inputs = autoencoder.inputs, outputs = autoencoder.get_layer('encoder').output)

# Query image from held out set where model never saw the query
random_idx = random.randint(1, test_data.shape[0])
query = test_data[random_idx]
#print(f'Query shape: {query.shape}')
##cv2.imshow("query", query)
##cv2.waitKey(0)

# Delete query image from the entire set
#print(f'Test data shape: {test_data.shape}')
new_testd = np.delete(test_data, random_idx, axis = 0)
#print(f'New Test data shape: {new_testd.shape}')
whole_data = np.concatenate([train_data, new_testd])
print(f'New Complete data shape: {whole_data.shape}')

# Get the latent space representation of the dataset
data_encoded = encoder.predict(whole_data)
query_encoded = encoder.predict(query.reshape(1,32,32,1)) # Have to reshape as explicitly one item since one entry is not interpreted correctly

#print(f'This is the data encoded shape {data_encoded.shape}')
#print(f'This is the query shape encoded {query_encoded.shape}')

# Use kNN to perform the query
neighbors = 5
kNN = NearestNeighbors(n_neighbors = neighbors)

# Train the kNN model on encoded dataset (with the query image deleted)
#kNN.fit(data_encoded.reshape(-1,4*4*8))
kNN.fit(data_encoded.reshape(-1,8*8*16)) # Reshape since kNN takes only 2 dim items
#distances, indices = kNN.kneighbors(query_encoded.reshape(1,4*4*8))
distances, indices = kNN.kneighbors(query_encoded.reshape(1,8*8*16))

print(f'These are the indices {indices}')
query_results = whole_data[indices]
query_results = query_results.reshape(-1,32,32,1)
print(f'This is the query results shape {query_results.shape}')

# Display the closest images
plt.figure('original image',figsize = (4,6))
plt.imshow(query.reshape(32,32,1))

plt.figure('closest images',figsize = (20,6))
for i in range(neighbors):
    ax = plt.subplot(1, neighbors, i+1)
    plt.imshow(query_results[i].reshape(32,32,1))

plt.show()
