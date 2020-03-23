#!/usr/bin/python
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from collections import Counter

#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics import mean_squared_error
#import xgboost as xgb
#from surprise import Reader, Dataset
#from surprise import BaselineOnly
#from surprise import KNNBaseline
#from surprise import SVD
#from surprise import SVDpp
#from surprise.model_selection import GridSearchCV


F_HR_MAT = "hr_mat.dat"
def load_hr_mat(): 
    ### build resource ID table
    return np.loadtxt(open(F_HR_MAT, "r"), delimiter=",", skiprows=0, dtype=int)

def build_sparse_mat(d):
    rid = d[:,1]-1   ## make rid start from 0
    smat = scipy.sparse.csr_matrix( ( d[:,2].astype(float), (d[:,0].astype(int), rid.astype(int))) )
    print ("Data shape: ", d.shape, smat.shape)
    print("Max/Min host: ", np.min(data[:,0]), np.max(data[:,0]) )
    print("Max/Min item: ", np.min(rid), np.max(rid) )          
    print("Max/Min data: ", np.min(data[:,2]), np.max(data[:,2]) )
    return smat

def eda():
    """
    Data report
    """
    data = load_hr_mat()
    print("Sparsity: %.6f" % (1 - len(data)/(np.max(data[:,0])*np.max(data[:,1]))     ) )
    print("# Hosts     : XXX", )
    print("# Resources : XXX", )
    plt.figure()
    plt.hist(data[:,0], bins=256)
    top = Counter(data[:,0]).most_common(1)
    plt.title("Histogram of # DL by each user - Top hid: %d" % top[0][0])
    plt.figure()
    plt.hist(data[:,1], bins=256)
    top = Counter(data[:,1]).most_common(1)
    plt.title("Histogram of # DL for each resource - Top rid: %d" % top[0][0])

def svd(train, k):
    utilMat = np.array(train)
    # the nan or unavailable entries are masked
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0)
    # nan entries will replaced by the average rating for each item
    utilMat = masked_arr.filled(item_means)
    x = np.tile(item_means, (utilMat.shape[0],1))
    # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x
    # The magic happens here. U and V are user and item features
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    # we take only the k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]
    UsV = np.dot(np.dot(U,s), V)
    UsV = UsV + x
    print("svd done")
    return UsV

eda()

#############
def test():
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1., 2., 3., 4., 5., 6.])
    A = scipy.sparse.csr_matrix((data, (row, col)) )
    print(A.todense())
    U, D, V = scipy.sparse.linalg.svds(A, k = 2)
    print(U.dot(np.diag(D).dot(V)))
##############

data = load_hr_mat()
smat = build_sparse_mat(data)

sddsf


###########################3
global_average = smat.sum()/smat.count_nonzero()
print ("Avg rate (over all u/m): ", global_average)


# Simple SVD approach for recommendation
U, sigma, Vt = scipy.sparse.linalg.svds(smat, k = 50)
smat1 = U.dot(np.diag(sigma)).dot(Vt)


idx=278
print(smat [idx,:])
print(np.argsort(smat2[idx,:])[::-1][:10] )


#===================== training
#train_sparse_matrix = sparse.csr_matrix((train_df["rating"].values, (train_df["user"].values,
#      train_df["movie"].values)),)
   
#us,mv = train_sparse_matrix.shape
#elem = train_sparse_matrix.count_nonzero()

#print("Sparsity Of Train matrix : {} % ".format(  (1-(elem/(us*mv))) * 100) )
