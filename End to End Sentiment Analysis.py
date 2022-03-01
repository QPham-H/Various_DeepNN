"""
End-to-end machine learning for Sentiment Analysis
Code by Quoc Pham
02.2022
"""

import numpy as np
import pandas as pd

# Constants for our datasets and model
BATCH = 64
EPOCHS = 1 # Temporarily one due to slow computer


# DATA COLLECTION
# Load the Amazon reviews dataset
amazon_df = pd.read_csv('amazon_alexa.tsv', sep='\t')

# EXPLORATORY DATA ANALYSIS
print(f'Amazon dataframe columns: {amazon_df.columns.values}')
print(f'Data info: {amazon_df.info()}')
print(amazon_df.head(5))

# Check for null values
print(amazon_df.isnull().sum())

# Check the distribution in the feedback column
label_dist = amazon_df['feedback'].value_counts()/len(amazon_df)
print(f'Label distribution: {label_dist}')

# DATA PREPARATION
# We can see that feedback is the data's binary indication of sentiment where 0=negative and 1=positive
# Create new columns by using a class to practice data preparation with a pipeline
import nltk
import re

from sklearn.base import TransformerMixin
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# Also create a class transformer to clean up the reviews
class ReviewPrep(TransformerMixin):
    def __init__(self, stopwords):
        self.stopwords = stopwords # Removing stopwords requires a new variable
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['verified_reviews'] = X['verified_reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stopwords)])) # reviews are idx = 3
        X['verified_reviews'] = X['verified_reviews'].apply(lambda x: re.sub('[^a-zA-Z\s]', '',x)) # Keep only letters
        return X

class TokenReview(TransformerMixin):
    def __init__(self, vocab, max_words):
        self.vocab = vocab
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words = self.vocab, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X['verified_reviews'])
        return self
    def transform(self, X):
        X['verified_reviews'] = self.tokenizer.texts_to_sequences(X['verified_reviews'])
        X['verified_reviews'] = pad_sequences(X['verified_reviews'], padding='post', maxlen=max_words)
        return X

#amazon_df['sentiment'] = amazon_df['sentiment'].apply(lambda x: 'positive' if x == 1 else 'negative')

# Set up a transformation pipeline for the data (even with one transformation, it is a good scalable machine learning practice)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def review_pipeline(stopwords, vocab, max_words):
    """
    Pipeline to prep the reviews and tokenize them

    Arguments:
        stopwords: stopwords to remove with ReviewPrep
        vocab: Vocab size for tokenizer
        max_words: max word length for padding
    Returns:
        text_pipeline: pipeline object of transformations for reviews
    """

    text_pipeline = Pipeline([
        ('cleanup', ReviewPrep(stopwords)),
        ('tokenize', TokenReview(vocab, max_words))
        ])
    return text_pipeline
    

def full_pipeline(data, stopwords, vocab, max_words):
    """
    Transformation pipeline for the data

    Arguments:
        data: original data
    Returns:
        prepped_data: fully transformed review data
        prepped_labels: review labels
    """

    
    all_transformers = ColumnTransformer([
        ('prep_rev', review_pipeline(stopwords, vocab, max_words), ['verified_reviews'])
        ])

    prepped_reviews = all_transformers.fit_transform(data)
    prepped_labels = data['feedback'].to_numpy()
    
    return prepped_reviews, prepped_labels

# Set stopwords to remove common but not useful words for analysis such as 'the', 'an'
stopwords = set(stopwords.words('english'))
vocab = 500 # The first x most used words aka vocab size
max_words = 20 # Max number of words in text (in order for Dense layer to connect to Embedding layer)
embedding_dim = 5 # Dimensions of vector to represent word in embedding layer

# Call the pipeline on the original dataset
prepped_reviews, prepped_labels = full_pipeline(amazon_df, stopwords, vocab, max_words)

# Quick check to see everything working
print(f'Prepped reviews sequence shape: {prepped_reviews.shape}')
print(f'Prepped labels shape: {prepped_labels.shape}')
print(f'Prepped data: {prepped_reviews}')


# BUILDING AND TRAINING THE MODELS

# Build a LSTM model
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Dropout, Input, Flatten

def build_LSTM(vocab=200, max_words=20, embedding_dim=8, neurons=20):
    """
    Builds an LSTM model for analysis

    Arguments:
        None
    Returns:
        model: an LSTM model
    """

    input_layer = Input(shape=(max_words))
    x = Embedding(vocab, embedding_dim, input_length=max_words)(input_layer) # Converts positive integer encoding of words as vectors into dimensional space where similiarity in meaning is represented by closeness in space
    #x = LSTM(64,dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x) # Maybe try relu for activation and output
    x = LSTM(neurons,dropout=0.1)(x)# Stacked LSTM potentially allows the hidden state to operate at different time scales
    #x = Flatten()(x)
    #x = Dense(16, activation='relu')(x)
    #x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(input_layer, out)

    model.compile(
    loss ='binary_crossentropy',
    optimizer='adam',
    metrics='acc'
    )
    
    return model

# Create the training and held-out test sets with stratified sampling so we better represent the data's proportions
from sklearn.model_selection import StratifiedShuffleSplit

# Fix random seed for reproducibility
seed = 33
np.random.seed(seed)

s_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed) # Using a random seed here for the Grid Search 

for train_idx, test_idx in s_split.split(prepped_reviews, prepped_labels):
    train_set = prepped_reviews[train_idx]
    train_labels = prepped_labels[train_idx]
    test_set = prepped_reviews[test_idx]
    test_labels = prepped_labels[test_idx]

# PICK THE BEST MODEL USING CROSS-VALIDATION
# In order to use sklearn on our model, first we have to wrap our model with KerasClassifier
from scikeras.wrappers import KerasClassifier

LSTM_wrapped = KerasClassifier(
    model=build_LSTM,
    neurons = 5,
    vocab = vocab,
    max_words = max_words,
    embedding_dim = embedding_dim,
    batch_size = BATCH,
    epochs = EPOCHS,
    random_state = seed,
    optimizer = 'adam',
    verbose = 0
    )

from sklearn.model_selection import cross_val_score

def cv_scores(model, train_set, train_labels):
    """
    Evaluate score by cross-validation

    Arguments:
        model: neural network model
        train_set: training set
    Returns:
        scores: an array of scores for each run
    """

    scores = cross_val_score(
        model,
        train_set,
        train_labels,
        scoring = 'accuracy',
        cv = 10
        )

    print(f'This is the average score of the cv: {scores.mean()}')
    return scores


LSTM_scores = cv_scores(LSTM_wrapped, train_set, train_labels)
print('CV Score for LSTM:')
print(LSTM_scores)


# HYPERPARAMETER TUNING
# Now that we've chosen the best model, we'll tune the hyperparameters
from sklearn.model_selection import GridSearchCV

# Define the grid search parameters
param_grid = [{
    'neurons' : [5, 10, 15, 20],
    'vocab': [500, 1000, 2000],
    'embedding_dim' : [5, 8, 10, 12]
    }]

grid = GridSearchCV(
    LSTM_wrapped,
    param_grid=param_grid,
    scoring = 'accuracy', # defaults to accuracy
    n_jobs = -1, # -1 means it'll use all the cores in the computer
    cv = 3 # defaults to 3
    )

grid_results = grid.fit(train_set, train_labels)

cv_scores = grid_results.cv_results_
for mean, params in zip(cv_scores['mean_test_score'],cv_scores['params']):
    print(f'Mean: {mean} with {params}')

print(f'Best score is: {grid_results.best_score_}')
print(f'Best params is: {grid_results.best_params_}')


# EVALUATE FINAL MODEL WITH HELD OUT TEST SET
# Best configuration
final_model = grid_results.best_estimator_

# Final evaluation
final_pred = final_model.predict(
    test_set
    )

final_acc = np.mean(test_labels == final_pred)

print(f'This is the test accuracy: {final_acc}')
