## Matrix Decompositions in Data Analysis
## Autumn 2024
## Assignment file
## FILL IN the following information
## Name:
## Student ID:

##
## This file contains stubs of code to help you to do your 
## assignment. You *must* fill your parts at the indicated positions
## and return this file as a part of your solution. 
##
## Remember to fill your name and student ID number above.
##
## This file is meant to be run with Python3

import numpy as np
from numpy.linalg import svd, norm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.stats import zscore

## In python, we can just write the code and it'll get called when the file is
## run with python3 assignment.py

## Task 1
##########

# Stub of a function that updates W and H using ALS
# All optimizer functions must have the same signature
# and compute one round of updates for W and H.
def nmf_als(A, W, H):
    # ADD YOUR code to update W and H
    return (W, H)

## Wrapper function for NMF
def nmf(A, k, optFunc=nmf_als, maxiter=300, repetitions=1):
    """Wrapper function for NMF.

    Computes NMF using optimizer function given as parameter. 
    This wrapper handles the iterations and multiple re-starts.

    Parameters:
    A -- input data
    k -- rank of the decomposition
    optFunc -- function that does one round of optimization for W and H
    maxiter -- maximum number of iterations (default = 300)
    repetition -- number of fresh re-starts (default = 1)
    """
    (n, m) = A.shape
    bestErr = np.Inf
    for rep in range(repetitions):
        # Init W and H 
        W = np.random.rand(n, k)
        H = np.random.rand(k, m)
        errs = [np.nan] * maxiter
        for i in range(maxiter):
            (W, H) = optFunc(A, W, H) # Call the actual optimizer given as parameter
            currErr = norm(A - np.matmul(W, H), 'fro')**2
            errs[i] = currErr
        if currErr < bestErr:
            bestErr = currErr
            bestW = W
            bestH = H
            bestErrs = errs
    return (bestW, bestH, bestErrs)




## Load the news data
A = np.genfromtxt('news.csv', delimiter=',', skip_header=1)
## To read the terms, just read the first line of news.csv
with open('news.csv') as f:
    header = f.readline()
    terms = [x.strip('"\n') for x in header.split(',')]

## Sample use of nmf_als with A
(W, H, errs) = nmf(A, 20, optFunc=nmf_als, maxiter=300, repetitions=1)
## To show the per-iteration error
plt.plot(errs)
plt.xlabel('Iterations')
plt.ylabel('Squared Frobenius')
plt.title('Convergence of NMF ALS')
plt.show()

## IMPLEMENT the other algorithms
## DO the comparisons



## Task 2
#########

## Normalise the data before applying the NMF algorithms
B = A/sum(sum(A)) # We're assuming Python3 here

## To print the top-10 terms of the first row of H, we can do the following
h = H[1,:]
ind = h.argsort()[::-1][:10]
for i in range(10): print("{}\t{}".format(terms[ind[i]], h[ind[i]]))

## USE NMF to analyse the data
## REPEAT the analysis with GKL-optimizing NMF

## Task 3
#########

## In Python, we can compute a slightly different normalized mutual information using scikit-learn's normalized_mutual_info_score (imported as nmi)
def nmi_news(x):
    gd = np.loadtxt('news_ground_truth.txt')
    return 1 - nmi(gd, x)

## We can compute Karhunen-Loeve 'manually'
Z = zscore(A)
U, S, V = svd(Z, full_matrices=False)
V.transpose()
V = V[0:20,:]
KL = np.matmul(Z, V.transpose())

## COMPUTE pLSA with the matrix B from the previous task

# Clustering the KL matrix
clustering = KMeans(n_clusters=20, n_init=20).fit(KL)
idx = clustering.labels_
## How good is this?
print("NMI for KL = {}".format(nmi_news(idx)))
