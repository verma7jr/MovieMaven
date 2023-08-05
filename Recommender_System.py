#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(2)


# In[4]:


credits.head(2)


# In[5]:


movies.shape


# In[6]:


credits.shape


# In[7]:


movies=movies.merge(credits ,on='title')


# In[8]:


movies.head(2)


# In[9]:


movies.shape


# In[10]:


movies.columns


# In[11]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[12]:


movies.head(2)


# In[13]:


movies.shape


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0]['genres']


# In[19]:


import ast #ast convert string to list

def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l    
    


# In[20]:


movies['genres']=movies['genres'].apply(convert)


# In[21]:


movies.iloc[0]['keywords']


# In[22]:


movies['keywords']=movies['keywords'].apply(convert)


# In[23]:


movies.head(2)


# In[24]:


movies.iloc[0]['cast']


# In[25]:


import ast #ast convert string to list

def convertcast(text):
    l=[]
    counter=0
    for i in ast.literal_eval(text):
        if counter<3:
            l.append(i['name'])
        counter+=1    
           
    return l  


# In[ ]:





# In[26]:


movies['cast']=movies['cast'].apply(convertcast)


# In[27]:


movies.head(2)


# In[28]:


movies.iloc[0]['crew']


# In[29]:


def fetchdirector(text):
    l=[]
    counter=0
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            l.append(i['name'])
            break         
           
    return l


# In[30]:


movies['crew']=movies['crew'].apply(fetchdirector)


# In[31]:


movies.head(2)


# In[32]:


movies.iloc[0]['overview']


# In[33]:


movies['overview']=movies['overview'].apply(lambda x:x.split())
movies.head(2)


# In[35]:


#Sam Worthington
#SamWorthington

def remove_space(word):
    l=[]
    for i in word:
        l.append(i.replace(" ",""))
    return l   
        


# In[36]:


movies['cast']=movies['cast'].apply(remove_space)
movies.head(2)


# In[37]:


movies['keywords']=movies['keywords'].apply(remove_space)
movies.head(2)


# In[38]:


movies['genres']=movies['genres'].apply(remove_space)
movies.head(2)


# In[39]:


movies['crew']=movies['crew'].apply(remove_space)
movies.head(2)


# In[40]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[41]:


movies.head(2)


# In[43]:


new_df=movies[['movie_id','title','tags']]


# In[44]:


new_df.head(2)


# In[45]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[46]:


new_df.head(2)


# In[47]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[48]:


new_df.head(2)


# In[50]:


import nltk
from nltk.stem import PorterStemmer


# In[51]:


ps=PorterStemmer()


# In[52]:


def stems(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)    


# In[53]:


new_df['tags']=new_df['tags'].apply(stems)


# In[54]:


new_df.iloc[0]['tags']


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[57]:


vector=cv.fit_transform(new_df['tags']).toarray()


# In[58]:


vector


# In[59]:


vector.shape


# In[61]:


from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


similarity=cosine_similarity(vector)


# In[64]:


similarity.shape


# In[65]:


new_df[new_df['title']=='Spider-Man'].index[0]


# In[67]:


def recommend(movie):
    index = new_df[new_df['title']==movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[68]:


recommend('Spider-Man')


# In[70]:


import pickle

pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[72]:


# pip install streamlit


# # In[74]:


# import streamlit as st
# import requests

# def fetch_poster(movie_id):
#     url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
#     data = requests.get(url)
#     data = data.json()
#     poster_path = data['poster_path']
#     full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
#     return full_path

# def recommend(movie):
#     index = movies[movies['title'] == movie].index[0]
#     distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
#     recommended_movie_names = []
#     recommended_movie_posters = []
#     for i in distances[1:6]:
#         # fetch the movie poster
#         movie_id = movies.iloc[i[0]].movie_id
#         recommended_movie_posters.append(fetch_poster(movie_id))
#         recommended_movie_names.append(movies.iloc[i[0]].title)

#     return recommended_movie_names,recommended_movie_posters


# st.header('Movie Recommender System Using Machine Learning')
# movies = pickle.load(open('movie_list.pkl','rb'))
# similarity = pickle.load(open('similarity.pkl','rb'))

# movie_list = movies['title'].values
# selected_movie = st.selectbox(
#     "Type or select a movie from the dropdown",
#     movie_list
# )

# if st.button('Show Recommendation'):
#     recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.text(recommended_movie_names[0])
#         st.image(recommended_movie_posters[0])
#     with col2:
#         st.text(recommended_movie_names[1])
#         st.image(recommended_movie_posters[1])

#     with col3:
#         st.text(recommended_movie_names[2])
#         st.image(recommended_movie_posters[2])
#     with col4:
#         st.text(recommended_movie_names[3])
#         st.image(recommended_movie_posters[3])
#     with col5:
#         st.text(recommended_movie_names[4])
#         st.image(recommended_movie_posters[4])

