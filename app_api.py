# -*- coding: utf-8 -*-
"""
Created on Sun May 30 15:41:08 2021

@author: Admin
"""
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import os
import pandas as pd
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser

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

session_dict = {
'dataPrediction' : pd.DataFrame()
}
latest_download_file = 'output1/Tweet_Prediction_Download.csv'

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
    df['pred'] = pred
    session_dict['dataPrediction'] = df.copy(deep=True)
    print(session_dict['dataPrediction'].shape)

    return df

###################################### API ####################################
from flask import send_file

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['SECRET_KEY'] =  os.urandom(24)

#GET API 
@app.route('/', methods = ['GET'])
@cross_origin(supports_credentials=True)
def get_index_page():
	return render_template('index.html')


#POST API with body parameters as filename
@app.route("/predict", methods=['POST'])
@cross_origin(supports_credentials=True)
def predict_sentiment_function():    
    filename = 'Output/test.csv'
    
    file = request.files['file']
    print(file)
    file.save(filename)
    dataset = pd.read_csv(filename)
    
    dataset = predict_sentiment(dataset)
    
    response = {
                'status': True,
                'responseData' : json.loads(dataset.to_json(orient='records'))
                }
    return jsonify(response)

@app.route('/downloadPrediction', methods=['GET'])
@cross_origin(supports_credentials=True)
def downloadPrediction():
    print('Enter to download latest data...')
    session_dict['dataPrediction'].to_csv(latest_download_file, index= False)
    print(session_dict['dataPrediction'].shape)
    return send_file(latest_download_file, attachment_filename= latest_download_file.split('/')[-1])

if __name__ == '__main__':
     app.run(port=5000)