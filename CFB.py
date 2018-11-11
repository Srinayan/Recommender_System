import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import time

data_matrix = pd.read_csv("data_matrix.csv")

movie_user_matrix = pd.read_csv("movie_user_matrix.csv")

movie_index = data_matrix.iloc[:,1:].columns
# print(movie_index)

sim_matrix = np.corrcoef(movie_user_matrix.iloc[1:,1:])
# print(sim_matrix.shape)
test_list = []

with open("test.txt", "rb") as fp:   # Unpickling
   test_list = pickle.load(fp)

movie_user_df = movie_user_matrix.iloc[1:,1:]
movie_user = movie_user_df.values

rmse = 0
start = time.time()
n = len(test_list)
relevant_list= []
m = np.sum(movie_user)
    # movie_user_matrix['sum_rows'] = movie_user_matrix.sum(axis=1)/6040     #along_columns
    # movie_user_matrix['sum_colums'] = movie_user_matrix.sum(axis=0)/3706    #along_rows
    # movie_user_matrix.to_csv("MV_sum.csv")
def BX(a,b):
    sum_row = movie_user[a].sum()
    sum_a = sum_row/6040
    sum_column = 0
    for i in movie_user:
        sum_column = sum_column+i[b]
    sum_b = sum_column/3706
    return sum_a+sum_b-m
for l in test_list:
    num = 0
    den = 0
    sim_vector = sim_matrix[l[0]]
    sim_movie_list=list(movie_index[(sim_vector>0.4)&(sim_vector<0.99999)])
    # print(len(sim_movie_list))
    if len(sim_movie_list) != 0:

        for movie in sim_movie_list:
            k = list(movie_index).index(movie)
            # print(movie)
            # print(k)
            # print(sim_vector[k],movie_user[k][l[1]])
            if movie_user[k][l[1]] != 0:
                num = num + (movie_user[k][l[1]]-BX(k,l[1]))*(sim_vector[k])
                den = den + sim_vector[k]
        if num==0 and den==0:
            predict_rating = movie_user[l[0]][l[1]]
        else:
            predict_rating = num/den + BX(l[0],l[1])

        # print("^^^^^^^^^^^^^^")
    else:
        predict_rating = movie_user[l[0]][l[1]]

    # print(l[0],l[1],predict_rating)
    rating = movie_user[l[0]][l[1]]
    rmse = (rating-predict_rating)**2 + rmse
    if predict_rating>3.0:
        relevant_list.append(predict_rating)

end = time.time()
# print(rmse,n)
RMSE = rmse**(0.5)
s_correlation =1-((6*rmse)/(n*(n**2-1)))
print("************")
print(RMSE)
print(s_correlation)
print(end - start)
print("************")
