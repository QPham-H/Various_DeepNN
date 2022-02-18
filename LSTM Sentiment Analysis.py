"""
Using three varieties of datasets to compile a general model for sentiment analysis
Code by Quoc Pham
02.2022
"""

import numpy as np
import pandas as pd

from nltk.corpus import stopwords

# Constants for our datasets and model
vocab = 5000 # Using the first x most used words aka vocab size
max_words = 500 # Max text size


# Load the IMDb dataset
imdb_df = pd.read_csv('IMDB_Dataset.csv')

print(f'IMDB dataframe columns: {imdb_df.columns}')    

# Load the Amazon reviews dataset
amazon_df = pd.read_csv('amazon_alexa.tsv',sep't')
amazon_df['sentiment'] = amazon_df.apply(lambda df: 'Positive' if df['rating'] >= 4 else 'Negative')  # axis 1 is to apply to each row

print(f'Amazon dataframe columns: {amazon_df.columns}')      

# Load the Twitter dataset
twitter_df = pd.read_csv('Sentiment.cvs')
print(f'Twitter dataframe columns: {twitter_df.columns}')

twitter_df.describe()

# Use only the two relevant columns
twitter_df = twitter_df[['text','sentiment']]
print(f'New twitter dataframe columns: {twitter_df.columns}')

# Combine all datasets to one dataframe
frames = [imdb_df,amazon_df,twitter_df]
merged_df = pd.concat(frames)

merged_df.head(10)

# Exploratory data analysis
merged_df.describe()

# Set stopwords to remove common but not useful words for analysis such as 'the', 'an'
stopwords = set(stopwords.words('english'))
merged_df['text'] = merged_df['text'].apply(lambda x: ''.join(x for x in x.split() if x not in stopwords))

merged_df.head(10)

# Remove digits, symbols, stopwords, and htmls
merged_df['text'] = merged_df['text'].str.lower()
merged_df['text'] = merged_df['text'].re.sub('[^a-zA-Z\s]', '') # Keep only letters 

merged_df.describe()




