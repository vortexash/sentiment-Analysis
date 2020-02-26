# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:50:26 2020

@author: trell
"""

import  pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import en_core_web_sm
from  spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()
import string
punctuations = string.punctuation
from spacy.lang.en import English
parser = English()
from sklearn.externals import joblib 
from transform import predictors


class training:
    def __init__(self):
        pass
    data_yelp = pd.read_table('yelp_labelled.txt')
    data_amazon = pd.read_table('amazon_cells_labelled.txt')
    data_imdb = pd.read_table('imdb_labelled.txt')
    combined_col= [data_amazon,data_imdb,data_yelp]
    for colname in combined_col:
        colname.columns = ["Review","Label"]
    company = [ "Amazon", "imdb", "yelp"]
    comb_data = pd.concat(combined_col,keys = company)
    stopwords = list(STOP_WORDS)
    def my_tokenizer(self,sentence):
        stopwords = list(STOP_WORDS)
        mytokens = parser(sentence)
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
        mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
        return mytokens
    classifier = LinearSVC()
  # Using Tfidf
    tfvectorizer = TfidfVectorizer()
  # Features and Labels
    X = comb_data['Review']
    ylabels = comb_data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)
  # Create the  pipeline to clean, tokenize, vectorize, and classify using"Count Vectorizor"
    pipe_countvect = Pipeline([("cleaner", predictors()),
                  ('vectorizer', tfvectorizer),
                  ('classifier', classifier)])

    pipe_countvect.fit(X_train,y_train)
obj=training()
joblib.dump(obj.pipe_countvect, 'sentiment_analysis.pkl') 