"""
Using three varieties of datasets to compile a general model for sentiment analysis
Code by Quoc Pham
02.2022
"""

import numpy as np
import pandas as pd
import nltk
import re

from bs4 import BeautifulSoup
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

# Constants for our datasets and model
vocab = 5000 # The first x most used words aka vocab size
max_words = 500 # Max number of words in text (in order for Dense layer to connect to Embedding layer) 
embedding_dim = 32 # Dimensions of vector to represent word in embedding layer
BATCH = 32
EPOCHS = 1

# Load the IMDb dataset
imdb_df = pd.read_csv('IMDB_Dataset.csv')

# Remove the html elements
def soup_get_text(text):
    soup = BeautifulSoup(text,'html.parser')
    return soup.get_text()

imdb_df['review'] = imdb_df['review'].apply(soup_get_text)

# Rename column headers for later merging
imdb_df.rename(columns={'review': 'text'}, inplace=True)

print(f'IMDB dataframe columns: {imdb_df.columns.values}')
print(imdb_df.head(5))

# Load the Amazon reviews dataset
amazon_df = pd.read_csv('amazon_alexa.tsv', sep='\t')

# Rename column headers and change to positive negative labels for later merging
amazon_df.rename(columns={'verified_reviews': 'text','feedback': 'sentiment'}, inplace=True)
amazon_df['sentiment'] = amazon_df['sentiment'].apply(lambda x: 'positive' if x == 1 else 'negative')

#amazon_df['sentiment'] = amazon_df.apply(lambda df: 'Positive' if df['rating'] >= 4 else 'Negative')  # axis 1 is to apply to each row

# Use only the two relevant columns
amazon_df = amazon_df[['text','sentiment']]
print(f'Amazon dataframe columns: {amazon_df.columns.values}')      
print(amazon_df.head(5))

# Load the Twitter dataset
twitter_df = pd.read_csv('Sentiment.csv')
#print(f'Twitter dataframe columns: {twitter_df.columns}')

# Use only the two relevant columns
twitter_df = twitter_df[['text','sentiment']]
twitter_df = twitter_df[twitter_df.sentiment != 'Neutral']

# Cleaning up the twitter data for merging later
twitter_df['sentiment'] = twitter_df['sentiment'].str.lower()

twitter_df['text'] = twitter_df['text'].apply(lambda x: re.sub(r'\bRT\b', '',x)) # Remove RT at the beginning of every tweet
print(f'New twitter dataframe columns: {twitter_df.columns.values}')
print(twitter_df.head(5))

# Combine all datasets to one dataframe
frames = [imdb_df,amazon_df,twitter_df]
merged_df = pd.concat(frames, ignore_index=True)

print(f'Merged dataframe columns: {merged_df.columns.values}')

# Set stopwords to remove common but not useful words for analysis such as 'the', 'an'
stopwords = set(stopwords.words('english'))
merged_df['text'] = merged_df['text'].str.lower() # making all text lowercase to match stopwords better
merged_df['text'] = merged_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# Remove digits, symbols, special characters, and htmls
merged_df['text'] = merged_df['text'].apply(lambda x: re.sub('[^a-zA-Z\s]', '',x)) # Keep only letters 

print(merged_df.head(5))

# Exploratory data analysis
print(merged_df.describe()) # Check that the sentiments labels have only 2 unique values

# Labelencoder to encode positive and negative into binary
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(merged_df['sentiment'].values)

# Tokenizer to encode words into a dictionary 
tokenizer = Tokenizer(num_words = vocab, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(merged_df['text'])

samples = tokenizer.texts_to_sequences(merged_df['text'])
samples = pad_sequences(samples, padding = 'post', maxlen = max_words)

# Create the training and test sets
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.2)


# Build the model
def build_model():
    input_layer = Embedding(vocab, embedding_dim, input_length=max_words) # Converts positive integer encoding of words as vectors into dimensional space where similiarity in meaning is represented by closeness in space
    x = LSTM(32,dropout=0.2, recurrent_dropout=0.2)(input_layer) # Try relu for activation and output
    out = Dense(1, activation='sigmoid')(x)

    model = Model(input_layer, out)

    print(model.summary())

    return model

build_model()

model.compile(
    loss ='binary_crossentropy',
    optimizer='adam',
    metrics='acc'
    )

history = model.fit(
    train_samples,
    train_labels,
    batch_size = BATCH,
    epochs = EPOCHS
    )

loss, accuracy = model.evaluate(
    test_samples,
    test_labels
    )

print(f'This is the test loss: {loss}')
print(f'This is the test accuracy: {accuracy}')
