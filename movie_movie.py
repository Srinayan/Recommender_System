import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


data_matrix = pd.read_csv("data_matrix.csv")

movie_index = data_matrix.iloc[:,1:].columns
print(len(movie_index))
print(movie_index[3705])

sim_matrix = np.corrcoef(data_matrix.T)

movie_user_matrix = data_matrix.T
movie_user_matrix.to_csv("movie_user_matrix.csv")

test_list=[]
for i in range(3337,3706):
    for j in range(6040):
        if movie_user_matrix.loc[movie_index[i],j] != 0:
            test_list.append([i,j])

with open("file.txt","w") as f:
    for item in test_list:
        f.write("%s\n" % item)

f.close()

with open("test.txt", "wb") as fp:   #Pickling
   pickle.dump(test_list, fp)


fp.close()
