# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


data=pd.read_excel('data.xlsx')
idea=data.iloc[:,0:1]

#idea=idea.set_index('Idea')
corpus=[]
for index,row in idea.iterrows():
    corpus.append(row['Idea'])

#test='Our idea is to be build an ML/AI based tool that helps Wolters Kluwer to understand what the deeper needs of end users. We are looking for a ML based tool that interacts with end users, learn about users needs, frame questions accordingly and find their deeper needs'
#corpus.append(test)



#Count Vectoriser then tidf transformer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
#vectorizer.get_feature_names()
#vectorizer.vocabulary_.get('attendance')
#print(X.toarray())     

#vectorizer.transform(['Something completely new.','Something not so new','Is this is how we do ?']).toarray()


#bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
#X_2 = bigram_vectorizer.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape )                        


from sklearn.metrics.pairwise import cosine_similarity

score=cosine_similarity(tfidf)
dist=1-score
print(dist)


from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf)

clusters = km.labels_.tolist()

idea={'Idea':corpus, 'Cluster':clusters}
frame=pd.DataFrame(idea,index=[clusters], columns=['Idea','Cluster'])
frame
frame['Cluster'].value_counts()

