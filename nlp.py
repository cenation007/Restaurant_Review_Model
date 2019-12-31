#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:38:54 2019

@author: ayush
"""
import pandas as pd
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning the texts
import re
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
corpus = []
for i in range(0,1000):
   review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) 
   review = review.lower()
   review = review.split()
   ps = PorterStemmer()
   review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
   review = ' '.join(review)
   corpus.append(review)
   
#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cd = CountVectorizer(max_features=1500)
X = cd.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
# Fitting classifier to the Training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)