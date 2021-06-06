# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:16:46 2021

@author: Admin
"""

import pandas as pd
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser
import pickle

html_parser = HTMLParser()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

apostrophe_dict = json.load(open('input/apostrophe_dict.json'))
short_word_dict = json.load(open('input/short_word_dict.json'))
emoticon_dict = json.load(open('input/emoticon_dict.json'))

vectorizer = pickle.load(open("output1/tfidf_vec_stem.pkl", "rb"))
with open('output1/tfidf_stem_RF.pkl', 'rb') as file:
    rf_model = pickle.load(file)


def remove_username(pattern, row):
    all_users = re.findall(pattern, row)
    for user in all_users:
        row = re.sub(user, '', row)
    return row

def convert_to_original_words(text, dictionary):
    for word in text.split():
        if word in dictionary:
            text = text.replace(word, dictionary[word])
    return text

def data_preprocessing(df):
    
    df['clean_tweet'] = df['tweet'].apply(lambda x: html_parser.unescape(x))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: remove_username('@[\w]*', x))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: x.lower())
    
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: convert_to_original_words(x, apostrophe_dict))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: convert_to_original_words(x, short_word_dict))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: convert_to_original_words(x, emoticon_dict))

    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))

    df['tweet_token'] = df['clean_tweet'].apply(lambda x: word_tokenize(x))
    df['tweet_token_filtered'] = df['tweet_token'].apply(lambda x: [word for word in x if not word in stop_words])
    
    df['tweet_stemmed'] = df['tweet_token_filtered'].apply(lambda x: ' '.join([stemmer.stem(i) for i in x]))
    df['tweet_lemmatized'] = df['tweet_token_filtered'].apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in x]))
    
    return df

def predict_sentiment(df):
    df = data_preprocessing(df)
    df_dtm = vectorizer.transform(df['tweet_stemmed'])
    pred = rf_model.predict(df_dtm)
    print(pred)
    return pred
  

df = pd.read_csv('data/test_tweets.csv')
df = df[:500]
print(df.shape)
print(df.head())

pred = predict_sentiment(df)
df['pred'] = pred
df.to_csv('output1/prediction.csv', index= False)