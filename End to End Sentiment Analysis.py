"""
End-to-end machine learning for Sentiment Analysis
Code by Quoc Pham
02.2022
"""

import numpy as np
import pandas as pd
import nltk
import re


# Constants for our datasets and model
vocab = 500 # The first x most used words aka vocab size
max_words = 20 # Max number of words in text (in order for Dense layer to connect to Embedding layer) 
embedding_dim = 5 # Dimensions of vector to represent word in embedding layer
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
amazon_df.isnull().sum()

# Check the distribution in the feedback column
label_dist = amazon_df['feedback'].value_counts()/len(amazon_df)
print(f'Label distribution: {label_dist}')

# DATA PREPARATION
# Rename column headers and change the feedback to positive/negative labels 
#amazon_df.rename(columns={'verified_reviews': 'text', inplace=True)

# We can see that feedback is the data's binary indication of sentiment where 0=negative and 1=positive
# Create new columns by using a class to practice data preparation with a pipeline
from sklearn.base import TransformerMixin
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

##class AddSentiments(TransformerMixin):
##    def fit(self, X, y=None):
##        return self
##    def transform(self, X):
##        sentiment = 'positive' if X[:,4] == 1 else 'negative' # Feedback column is idx = 4
##        return np.c_[X, sentiment]

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
    

def pipeline(data, stopwords, vocab, max_words):
    """
    Transformation pipeline for the data

    Arguments:
        data: original data
    Returns:
        prepped_data: fully transformed review data
        prepped_labels: review labels
    """

##    review_pipeline = Pipeline([
##        ('add_sent', AddSentiment())
##        ])

    
    all_transformers = ColumnTransformer([
##        ('add_sent', AddSentiment(), ['feedback']),
        ('prep_rev', review_pipeline(stopwords, vocab, max_words), ['verified_reviews'])
        ])

    prepped_reviews = all_transformers.fit_transform(data)
##    print(f'prepped review shape :{prepped_reviews.shape}')
##    print(f'prepped reviews :{prepped_reviews}')

    prepped_labels = data['feedback'].to_numpy()
    
    return prepped_reviews, prepped_labels

# Set stopwords to remove common but not useful words for analysis such as 'the', 'an'
stopwords = set(stopwords.words('english'))

# Call the pipeline on the original dataset
prepped_reviews, prepped_labels = pipeline(amazon_df, stopwords, vocab, max_words)

# Quick check to see everything working
print(f'Prepped data sequence: {prepped_reviews.shape}')
print(f'Prepped labels: {prepped_labels.shape}')      

### Labelencoder to encode positive and negative into binary
##labelencoder = LabelEncoder()
##labels = labelencoder.fit_transform(merged_df['sentiment'].values)

# Tokenizer to encode words into a dictionary 
##tokenizer = Tokenizer(num_words = vocab, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
##tokenizer.fit_on_texts(merged_df['text'])
##
##samples = tokenizer.texts_to_sequences(merged_df['text'])
##samples = pad_sequences(samples, padding = 'post', maxlen = max_words)
##print(samples)

# BUILDING AND TRAINING THE MODELS
# Create the training and held-out test sets with stratified sampling so we better represent the data's proportions
from sklearn.model_selection import StratifiedShuffleSplit

# Fix random seed for reproducibility
seed = 33
np.random.seed(seed)

s_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed) # Using a random seed here for the Grid Search 
train_idx, test_idx =  s_split.split(prepped_reviews, prepped_labels)

for train_idx, test_idx in zip(train_idx,test_idx):
    train_set = processed_df.loc[train_idx]
    test_set = processed_df.loc[test_idx]

# Check training distribution
train_dist = train_set.value_counts() / len(train_set)
print(f'Training distribution: {train_dist}')

#train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.2)


#print(train_samples)

# Build a LSTM model
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Dropout, Input, Flatten
##from sklearn.base import BaseEstimator
##
##class LSTM_model(BaseEstimator):
##    def __init__(self, vocab, max_words, embedding_dim):
##        self.vocab = vocab
##        self.max_words = max_words
##        self.embedding_dim = embedding_dim
##        self.model = build_LSTM(self.vocab,self.max_words,self.embedding_dim)
##    def fit(self, 

def build_LSTM(vocab, max_words, embedding_dim, neurons=5):
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
    x = Flatten()(x)
    #x = Dense(16, activation='relu')(x)
    #x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(input_layer, out)

    print(model.summary())
    return model

def compile_fit(model, train_set):
    """
    Compiles and fit the model on training data

    Arguments:
        model: neural network model
        train_set: training set
    Returns:
        None
    """

    model.compile(
        loss ='binary_crossentropy',
        optimizer='adam',
        metrics='acc'
        )

    history = model.fit(
        train_set[:,0],
        train_set[:,1],
        batch_size = BATCH,
        epochs = EPOCHS
        )

##def evaluate_model(model, test_set):
##    """
##    Compiles and fit the model on training data
##
##    Arguments:
##        model: neural network model
##        test_set: test set
##    Returns:
##        loss: loss for the test set
##        accuracy: accuracy for the test set
##    """
##
##    loss, accuracy = model.evaluate(
##        test_set[:,0],
##        test_set[:,1]
##        )
##
##    print(f'This is the test loss: {loss}')
##    print(f'This is the test accuracy: {accuracy}')
##    return loss, accuracy

LSTM = build_LSTM()
compile_fit(LSTM, train_set)
#loss, accuracy = evaluate_model(LSTM, test_set)

# PICK THE BEST MODEL USING CROSS-VALIDATION
from sklearn.model_selection import cross_val_score

def cv_scores(model, train_set):
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
        train_set[:,0],
        train_set[:,1],
        scoring = 'accuracy',
        cv = 10
        )

    print(f'This is the average score of the cv: {scores.mean()}')
    return scores

print(f'CV Score for LSTM:')
LSTM_scores = cv_scores(LSTM, train_set)

print(throwerror)

# HYPERPARAMETER TUNING
# Now that we've chosen the best model, we'll tune the hyperparameters
# In order to use sklearn's GridSearchCV function for testing hyperparameters, we first have to wrap our model with KerasClassifier
from tf.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

model = KerasClassifier(
    build_fn=build_LSTM,
##    vocab=vocab,
##    max_words=max_words,
##    embedding_dim=embedding_dim,
    batch_size = BATCH,
    epochs = EPOCHS
    )

# Define the grid search parameters
param_grid = [{
    'neurons' : [5, 10, 15, 20],
    'vocab': [500, 1000, 2000],
    'max_words' : [20, 40, 80],
    'embedding_dim' : [5, 8, 10, 12]
    }]

grid = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring = 'accuracy', # defaults to accuracy
    n_jobs = -1, # -1 means it'll use all the cores in the computer
    cv = 3 # defaults to 3
    )

grid_results = grid.fit(train_set[:,0], train_set[:,1])

print(f'Best score is: {grid_results.best_score_}')
print(f'Best params is: {grid_results.best_params_}')

cv_scores = grid_results.cv_results_
for mean, params in zip(cv_scores['mean_test_score'],cv_scores['params']):
    print(f'Mean: {mean} with {params}')

# EVALUATE FINAL MODEL WITH HELD OUT TEST SET
# Best configuration
final_model = grid_results.best_estimator_

# Final evaluation
loss, accuracy = final_model.evaluate(
    test_set[:,0],
    test_set[:,1]
    )

print(f'This is the test loss: {loss}')
print(f'This is the test accuracy: {accuracy}')

