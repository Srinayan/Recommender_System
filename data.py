import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

movies_data = pd.read_table('movies.dat', header=None, sep='::', names=['movie_id', 'title', 'genre'],engine='python')
movies_data.to_csv("movies_data.csv")


ratings_data = pd.read_table('ratings.dat',header=None, sep='::', names=['user_id', 'movie_id', 'rating','timestamp'],engine='python')
del ratings_data['timestamp']


data_table = pd.merge(ratings_data,movies_data,on='movie_id')[['user_id','title','movie_id','rating']]
data_matrix = data_table.pivot_table(values='rating',index='user_id', columns='title')
data_matrix.fillna(0,inplace=True)
data_matrix.to_csv("data_matrix.csv")
