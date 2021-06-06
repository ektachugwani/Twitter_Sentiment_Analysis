# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:45:09 2021

@author: Admin
"""
import pandas as pd
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from html.parser import HTMLParser


df = pd.read_csv('data/train_tweet.csv') #31962
print(df.head())

#label '1' denotes the tweet is racist/sexist and
#label '0' denotes the tweet is not racist/sexist
print(df.label.value_counts()) #29720:0, 2242:1

###############################################################################
################################ Preprocessing ################################
###############################################################################

#1 Use html parser to remove html format as &lt; with <
html_parser = HTMLParser()
df['clean_tweet'] = df['tweet'].apply(lambda x: html_parser.unescape(x))

#-------------------------------------------#
#2 Remove @username 
def remove_username(pattern, row):
    all_users = re.findall(pattern, row)
    for user in all_users:
        row = re.sub(user, '', row)
    return row
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: remove_username('@[\w]*', x))

#-------------------------------------------#
#3 Convert text into lowercase
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: x.lower())

#-------------------------------------------#
#4 Replace apostrophe or short words or emojis with original words
#check count of apostrophe or short words or emojis available in column
def word_count(col, dictionary):
    count_dict ={}
    for key, val in dictionary.items():
        for x in col:
            if key in x:
                if key in count_dict.keys():
                    count_dict[key] = count_dict[key]+1
                else:
                    count_dict[key] = 1
    return count_dict

# function to replace as original words from apostrophe or short words
def convert_to_original_words(text, dictionary):
    for word in text.split():
        if word in dictionary:
            text = text.replace(word, dictionary[word])
    return text


apostrophe_dict = json.load(open('input/apostrophe_dict.json'))
count_dict = word_count(df['clean_tweet'], apostrophe_dict)
print('\nCount of apostrophe words : \n', count_dict)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: convert_to_original_words(x, apostrophe_dict))
print(df['clean_tweet'].iloc[7])


short_word_dict = json.load(open('input/short_word_dict.json'))
count_dict = word_count(df['clean_tweet'], short_word_dict)
print('\nCount of short words : \n', count_dict)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: convert_to_original_words(x, short_word_dict))
print(df['clean_tweet'].iloc[228])


emoticon_dict = json.load(open('input/emoticon_dict.json'))
count_dict = word_count(df['clean_tweet'], emoticon_dict)
print('\nCount of emoticon : \n', count_dict)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: convert_to_original_words(x, emoticon_dict))
print(df['clean_tweet'].iloc[128])

#-------------------------------------------#
#plot most used hashtags
#check hastags used mostly
def hashtag_extract(x):
    hashtags = []
    
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.extend(ht)

    return hashtags

HT_regular = hashtag_extract(df['clean_tweet'][df['label'] == 0])
HT_negative = hashtag_extract(df['clean_tweet'][df['label'] == 1])

top_negative_hashtags = pd.Series(HT_negative).value_counts().sort_values(ascending=False).head(20)
top_negative_hashtags.plot(kind='bar')

top_regular_hashtags = pd.Series(HT_regular).value_counts().sort_values(ascending=False).head(20)
top_regular_hashtags.plot(kind='bar')


#-------------------------------------------#
#6 Remove punctuations
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))

#-------------------------------------------#
#7 Replace special characters
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))

#-------------------------------------------#
#8 Replace numbers
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))

'''
#-------------------------------------------#
#9 correct spell check 
from textblob import TextBlob
df['correct_tweet'] = df['clean_tweet'].apply(lambda x: str(TextBlob(x).correct()))
df.to_csv('output/corrected_tweet.csv', index= False)

'''
df.to_csv('output/cleaned_tweet.csv', index= False)

######################### Stemming and Lemmatization ##########################

df = pd.read_csv('output/cleaned_tweet.csv')

#-------------------------------------------#
#10 tokenize words and remove stopwords
# Creating token for the clean tweets
df['tweet_token'] = df['clean_tweet'].apply(lambda x: word_tokenize(x))
stop_words = set(stopwords.words('english'))
df['tweet_token_filtered'] = df['tweet_token'].apply(lambda x: [word for word in x if not word in stop_words])

#-------------------------------------------#
#11 Stemming - remove ing, ly,etc from words
stemmer = PorterStemmer()
df['tweet_stemmed'] = df['tweet_token_filtered'].apply(lambda x: ' '.join([stemmer.stem(i) for i in x]))

#-------------------------------------------#
#12 Lemmatization - convert words in base form
lemmatizer = WordNetLemmatizer()
df['tweet_lemmatized'] = df['tweet_token_filtered'].apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in x]))

#df.to_csv('output/preprocessed_data_corrected_tweet.csv', index= False)
df.to_csv('output/preprocessed_data.csv', index= False)


###############################################################################
################################ Wordcloud ####################################
###############################################################################

def plot_word_cloud(col, title, filepath):
    words = ' '.join([text for text in col])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

plot_word_cloud(df['tweet_stemmed'], 'Common words in stemmed', 'output/all_stemmed_corrected.png')
plot_word_cloud(df['tweet_lemmatized'], 'Common words in lemmatized', 'output/all_lemmatized_corrected.png')

#non-racist words
plot_word_cloud(df['tweet_stemmed'][df['label']==0], 'Non-racist words in stemmed', 'output/non_racist_stemmed_corrected.png')
plot_word_cloud(df['tweet_lemmatized'][df['label']==0], 'Non-racist words in lemmatized', 'output/non_racist_lemmatized_corrected.png')

#racist words
plot_word_cloud(df['tweet_stemmed'][df['label']==1], 'Racist words in stemmed', 'output/racist_stemmed_corrected.png')
plot_word_cloud(df['tweet_lemmatized'][df['label']==1], 'Racist words in lemmatized', 'output/racist_lemmatized_corrected.png')


###############################################################################
############################## Feature Extraction #############################
###############################################################################

import itertools

def plot_confusion_matrix(cm,filename, classes,normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    plot_filename = filename
    fig, ax = plt.subplots(figsize = (6,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.grid(False) 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
        


import numpy as np
df = pd.read_csv('output/preprocessed_data_corrected_tweet.csv')
#df = pd.read_csv('output/preprocessed_data.csv')
df = df.replace(np.nan, '', regex=True)


count_vec_stem = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english').fit(df['tweet_stemmed'])
count_vec_stemm = count_vec_stem.transform(df['tweet_stemmed'])

count_vec_lemm = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english').fit(df['tweet_lemmatized'])
count_vec_lemmm = count_vec_lemm.transform(df['tweet_lemmatized'])

tfidf_vec_stem = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english').fit(df['tweet_stemmed'])
tfidf_vec_stemm = tfidf_vec_stem.transform(df['tweet_stemmed'])

tfidf_vec_lemm = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english').fit(df['tweet_lemmatized'])
tfidf_vec_lemmm = tfidf_vec_lemm.fit_transform(df['tweet_lemmatized'])

import pickle

#pickle.dump(tfidf_vec_stem, open("output1/tfidf_vec_stem_corrected.pkl", "wb"))


df_metrics = []

def train_model(vectorizer, estimator, vect_name, est_name):
    X, x_test, Y, y_test = train_test_split(vectorizer, df['label'], random_state=42, test_size=0.3)
    estimator.fit(X, Y) # training the model
    y_pred = estimator.predict(x_test)    
    #y_proba = estimator.predict_proba(x_test)    
    #roc_auc = roc_auc_score(y_test, y_proba[:,1])
    
    score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, filename='output/'+vect_name+est_name+'.png' ,classes= [0,1], title='Confusion matrix '+ est_name)

    
    f1_scor = f1_score(y_test, y_pred)
    print('Confusion matrix : ', cm)
    
    data = {'estimator': est_name, 'vectorizer': vect_name, #'roc_auc' :roc_auc,
            'accuracy_score': score, 'F1-score':f1_scor}
    df_metrics.append(data)
    
    '''
    if vect_name == 'tfidf_vec_stem' and est_name == 'Random Forest Classifier':
        with open('output1/tfidf_stem_RF.pkl', 'wb') as file:  
            pickle.dump(estimator, file)
    '''
    return data


vect_dict = {'count_vec_stem' : count_vec_stemm, 'count_vec_lemm' : count_vec_lemmm,
    'tfidf_vec_stem' : tfidf_vec_stemm, 'tfidf_vec_lemm' : tfidf_vec_lemmm}

from sklearn.svm import SVC

estimator = LogisticRegression()
estimator1 = RandomForestClassifier()
estimator2 = SVC()

for name, vect in vect_dict.items() : 
    train_model(vect, estimator, name, 'Logistic Regression')
    train_model(vect, estimator1, name, 'Random Forest Classifier')
    train_model(vect, estimator2, name, 'Support Vector Machine')

pd.DataFrame(df_metrics).to_csv('output/Evaluation_metrics_corrected_tweet.csv', index= False)
#pd.DataFrame(df_metrics).to_csv('output1/Evaluation_metrics.csv', index= False)









#https://arxiv.org/pdf/1808.10245v1.pdf
#http://www.cs.columbia.edu/~julia/papers/Agarwaletal11.pdf
#https://github.com/saadbinmanjur/Kaggle-Competition-Sentiment-Analysis-on-Twitter-tweets/blob/main/Sentiment%20Analysis%20on%20Twitter%20tweets.ipynb

#https://www.kaggle.com/gauravchhabra/nlp-twitter-sentiment-analysis-project