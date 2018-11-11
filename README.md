# Recommender_System
Dataset we have have used is movielens which contain the ratings giving by user for a particular movie. The size of our dataset is 3706 by 6040

Data.py : Data.py file contains code which transforms the dataset from https://grouplens.org/datasets/movielens/ to sparse matrix named data_matrix with rows as users and items as columns. 

Movie_movie.py : we construct similarity matrix named sim_matix by taking transpose of original matrix . Mij in sim_matrix mean similarity of item i with j . test_list contains the values whose rating has to be predictated.

CF.py : we traverse through test_list and predict the values of items and calculate rmse , s_correlation and time .

CFB.py : we traverse through test_list and predict the values of items along with adding baseline index and add it to the predicted rating  and calculate rmse , s_correlation and time. 


SVD.py : Taking the movie_user.csv file as matrix (which is sparse ) and decomposing the matrix using svd decomposition algorithm . calculate rmse , s_correlation and time. 

CUR.py :  Taking the movie_user.csv file as matrix (which is sparse ) and decomposing the matrix using svd and then multiplying C(column) U (calculated using svd decomposition) decomposition algorithm and R(rows) . number chosen for random selection of rows and columns is rank of our matrix. Then calculate rmse , s_correlation and time. 
