#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


df = pd.read_csv("movie_dataset.csv")
df.columns


# In[6]:


features = ['keywords','cast','genres','director']


# In[7]:


def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']


# In[8]:


for feature in features:
    df[feature] = df[feature].fillna('')
df["combined_features"] = df.apply(combine_features,axis=1)


# In[9]:


cv = CountVectorizer() 
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)


# In[10]:


def find_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def find_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# In[11]:


movie = input("Enter movie name of your choice: ")
movie_index = find_index_from_title(movie)


# In[12]:


similar_movies = list(enumerate(cosine_sim[movie_index]))


# In[13]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


# In[14]:


i=0
for element in sorted_similar_movies:
    print(find_title_from_index(element[0]))
    i=i+1
    if i>5:
        break

