# -*- coding: utf-8 -*-
''' This program takes a excel sheet as  '''

import pandas as pd
import numpy as np


data=pd.read_excel('data.xlsx') #Include your data file instead of data.xlsx
idea=data.iloc[:,0:1] #Selecting the first column that has text.

#Converting the column of data from excel sheet into a list of documents, where each document corresponds to a group if sentences.
corpus=[]
for index,row in idea.iterrows():
    corpus.append(row['Idea'])

#Count Vectoriser then tidf transformer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

#vectorizer.get_feature_names()

#print(X.toarray())     

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape )                        

from sklearn.cluster import KMeans

num_clusters = 5 #Change it according to your data.
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()

idea={'Idea':corpus, 'Cluster':clusters} #Creating dict having doc with the corresponding cluster number.
frame=pd.DataFrame(idea,index=[clusters], columns=['Idea','Cluster']) # Converting it into a dataframe.

print("\n")
print(frame) #Print the doc with the labeled cluster number.
print("\n")
print(frame['Cluster'].value_counts()) #Print the counts of doc belonging to ach cluster.

