# -*- coding: utf-8 -*-
''' This program takes a excel sheet as input where each row in first column of sheet represents a document.  '''

import pandas as pd
import numpy as np


data=pd.read_excel('data.xlsx') #Include your data file instead of data.xlsx
idea=data.iloc[:,0:1] #Selecting the first column that has text.

#Converting the column of data from excel sheet into a list of documents, where each document corresponds to a group of sentences.
corpus=[]
for index,row in idea.iterrows():
    corpus.append(row['Idea'])

'''Or you could just comment out the above code and use this dummy corpus list instead if you don't have the data.

corpus=['She went to the airport to see him off.','I prefer reading to writing.','Los Angeles is in California. It's southeast of San Francisco.','I ate a burger then went to bed.','Compare your answer with Tom's.','I had hardly left home when it began to rain heavily.','If he had asked me, I would have given it to him. 
','I could have come by auto, but who would pay the fare? ','Whatever it may be, you should not have beaten him.','You should have told me yesterday','I should have joined this course last year.','Where are you going?','There are too many people here.','Everyone always asks me that.','I didn't think you were going to make it.','Be quiet while I am speaking.','I can't figure out why he said so.'] '''
    
    
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
print(frame['Cluster'].value_counts()) #Print the counts of doc belonging to each cluster.

