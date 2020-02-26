# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:09:39 2020

@author: Ashish Lepcha
"""

from sklearn.externals import joblib 
class Sentiment_analysis:
    def __init__(self):
        pass
    @staticmethod
    def sentiments(data):
        model = joblib.load('sentiment_analysis.pkl') 
        return model.predict(data) 

oj=Sentiment_analysis()
example = ["sucks, the product is fucked up",
 "My video takes too much time to load",
 "I feel amazing!"]
print(oj.sentiments(example))
