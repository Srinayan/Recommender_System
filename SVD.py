import csv
import pandas as pd
import numpy as np
from numpy import linalg
import time

matrix_A = pd.read_csv("movie_user_matrix.csv")
matrix_A = matrix_A.iloc[1:,1:]
# print(matrix_A)
matrix_AT = matrix_A.T
A = matrix_A.values
AT = matrix_AT.values
n = 3706*6040 #size of the matrix
start = time.time()
A_AT = np.dot(A,AT)
U_eigen_values,U_eigen_vectors = linalg.eig(A_AT)
U_eigen_values = np.absolute(U_eigen_values.real)
U_eigen_vectors = U_eigen_vectors.real
# print(eigen_values)
# print(eigen_vectors)
def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q
U = gram_schmidt(U_eigen_vectors)
AT_A = np.dot(AT,A)
# print(AT_A.shape)
V_eigen_values,V_eigen_vectors = linalg.eig(AT_A)
V_eigen_values = np.absolute(V_eigen_values.real)
V_eigen_vectors = V_eigen_vectors.real

V = gram_schmidt(V_eigen_vectors)
VT = np.transpose(V)
eigen_values = np.sqrt(U_eigen_values)
eigen_values = np.sort(eigen_values)
S =np.zeros((3706,6040))
np.fill_diagonal(S,eigen_values)
print(S.shape)
print(U.shape,S.shape,VT.shape)
X = np.dot(U,S)
SVD = np.dot(X,VT)
end = time.time()

rmse_M = np.subtract(A,SVD)
rmse = rmse_M**2
RMSE_M = rmse.sum()
RMSE_value = RMSE_M**0.5
s_correlation = 1-((6*RMSE_M)/(n*(n**2-1)))
print("***********")
print(RMSE_value)
print(s_correlation)
print(end-start)
print("***********")
