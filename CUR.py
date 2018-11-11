import csv
import pandas as pd
import numpy as np
from numpy import linalg

matrix_A = pd.read_csv("movie_user_matrix.csv")
matrix_A = matrix_A.iloc[1:,1:]
A = matrix_A.values
#
r = np.linalg.matrix_rank(matrix_A)
#
# C = matrix_A.sample(r,axis=1)
# C.to_csv("C.csv")
# R = matrix_A.sample(r,axis=0)
# R.to_csv("R.csv")

C = pd.read_csv("C.csv",header=None)
R = pd.read_csv("R.csv",header=None)
selected_C = C.iloc[0:,1:]
inter_list=selected_C.values[0]
# print(inter_list)
CM = C.iloc[1:,2:]
# print(CM.shape)
RM = R.iloc[2:,1:]
# print(RM.shape)
inter_matrix = R.iloc[:,inter_list]
matrix_W = inter_matrix.iloc[2:,1:]

matrix_WT =matrix_W.T

W = matrix_W.values
WT = matrix_WT.values

W_WT = np.dot(W,WT)
X_eigen_values,X_eigen_vectors = linalg.eig(W_WT)
X_eigen_values = np.absolute(X_eigen_values.real)
X_eigen_vectors = X_eigen_vectors.real
# print(X_eigen_values)
# print(eigen_values)
# print(eigen_vectors)
def gram_schmidt(M):
    Q, R = np.linalg.qr(M)
    return Q
X = gram_schmidt(X_eigen_vectors)
XT = np.transpose(X)
WT_W = np.dot(WT,W)
Y_eigen_values,Y_eigen_vectors = linalg.eig(WT_W)
Y_eigen_values = np.absolute(Y_eigen_values.real)
Y_eigen_vectors = Y_eigen_vectors.real

Y = gram_schmidt(Y_eigen_vectors)
# print("eigen_values")
eigen_values = np.sqrt(X_eigen_values)
print(XT.shape)
print(Y.shape)
Z =np.zeros((3662,3662))

np.fill_diagonal(Z,eigen_values)

ZP= np.linalg.pinv(Z)
# print("PINV")

ZP_2 = np.dot(ZP,ZP)

Y_Z = np.dot(Y,ZP_2)

U = np.dot(Y_Z,XT)

CUR_1 = np.dot(CM,U)
CUR = np.dot(CUR_1,RM)

end = end.time()
n = 3706*6040
rmse_M = np.subtract(A,CUR)
rmse = rmse_M**2
RMSE_M = rmse.sum()
RMSE_value = RMSE_M**0.5
s_correlation = 1-((6*RMSE_M)/(n*(n**2-1)))
print("***********")
print(RMSE_value)
print(s_correlation)
print(end-start)
print("***********")
