## ===================================================================
## Matrix Decompositions in Data Analysis
## Autumn semester 2024
##
## Assignment 2 - NMF

## Student's information
## Name: Hoang Pham
## Email: hpham@uef.fi
## Student ID #: 2417385
## ===================================================================


# libraries
import os
import math
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from numpy.linalg import svd, norm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import NMF
from scipy.stats import zscore

## ===================================================================
## TASK 1: ALS vs. multiplicative NMF
## ===================================================================

# ========== NMF Algorithm (ALS, Multiplicative and GD OPL) ==========
# NMF-ALS update
def nmf_als(A, W, H):
    # Compute pseudo-inverse of W
    W_pinv = np.linalg.pinv(W)

    # Update H
    H = np.matmul(W_pinv, A)
    H = np.maximum(H, 0)

    # Compute pseudo-inverse of W
    H_pinv = np.linalg.pinv(H)

    # Update W
    W = np.matmul(A, H_pinv)
    W = np.maximum(W, 0)
    return (W, H)


# NMF-Multiplicative update
def nmf_multiplicative(A, W, H, epsilon=0.0001):
    """
        W : [n, k]
        H : [k, m]
    """

    H = H * (np.dot(W.T, A) / (np.dot(np.dot(W.T, W), H) + epsilon))
    W = W * (np.dot(A, H.T) / (np.dot(np.dot(W, H), H.T) + epsilon))

    return (W, H)


# NMF-GD-OPL update
def nmf_gd_opl(A, W, H):
    rowSums = np.sum(np.dot(W.T, W), axis=1)
    n_w = np.diag(1 / rowSums) # 

    G_w = np.dot(np.dot(W.T, W), H) - np.dot(W.T, A)
    H = H - np.dot(n_w, G_w)
    H = np.maximum(H, 0)

    return (W, H)


# ======== Task 1 utils (compute nmf, plot figure, find converge points) =========
# NMF running
def nmf(A, k, optFunc=nmf_als, maxiter=300, repetitions=1):
    (n, m) = A.shape
    bestErr = np.Inf
    bestErrors = None
    running_times = None

    errs_multi_repeats = []
    running_multi_repeats = []

    for rep in range(repetitions):
        print("Repetition {}".format(rep+1))

        # Init W and H
        W = np.random.rand(n, k)
        H = np.random.rand(k, m)
        errs = [np.nan] * maxiter
        rtimes = [np.nan] * maxiter

        start_time = time.time()
        for i in tqdm(range(maxiter)):
            (W, H) = optFunc(A, W, H) # Call the actual optimizer given as parameter
            currErr = norm(A - np.matmul(W, H), 'fro')**2
            errs[i] = currErr
            check_point_time = time.time()
            rtimes[i] = check_point_time - start_time

            if currErr < bestErr:
                bestErr = currErr
                bestW = W
                bestH = H
                bestErrors = errs
                running_times = rtimes

        errs_multi_repeats.append(errs)
        running_multi_repeats.append(rtimes)
    if repetitions == 1:
        return (bestW, bestH, bestErrors, running_times)
    else:
        return (bestW, bestH, errs_multi_repeats, running_multi_repeats)
    
    
# Run NMF with multiple restarts
def nmf_multi_repeats(A, repetitions=10):
    errs_hist = []
    running_times_hist = []
    for _ in range(repetitions):
        (W_als, H_als, errs_als, running_times_als) = nmf(A, 20, optFunc=nmf_als, maxiter=100, repetitions=repetitions)
        errs_hist.append(errs_als)
        running_times_hist.append(running_times_als)
    return (errs_hist, running_times_hist)


# Find the converge points of reconstruction errors
def find_converge_points(errs, T1=0.0001, T2=0.00001, T3=0.000001):
    errs = np.array(errs)
    errs_diff = abs(errs[1:] - errs[:-1])/errs[:-1]
    converge_points = [0] * 3
    errs_diff_t1 = errs_diff < T1
    errs_diff_t2 = errs_diff < T2
    errs_diff_t3 = errs_diff < T3

    s1, s2, s3 = 1, 1, 1
    for i in range(1, len(errs_diff)):
        if errs_diff_t1[i]:
            errs_diff_t1[i] = s1
            s1 += 1
        else:
            s1 = 1
        if s1 == 10:
            converge_points[0] = i - 9 + 1

        if errs_diff_t2[i]:
            errs_diff_t2[i] = s2
            s2 += 1
        else:
            s2 = 1
        if s2 == 10:
            converge_points[1] = i - 9 + 1

        if errs_diff_t3[i]:
            errs_diff_t3[i] = s3
            s3 += 1
        else:
            s3 = 1
        if s3 == 10:
            converge_points[2] = i - 9 + 1
    return converge_points
    

# Find the converge points following running time
def find_converge_points_time(errors, running_times, converge_points):
    converge_points_time = [0] * 3
    for i, cp in enumerate(converge_points):
        converge_points_time[i] = running_times[cp]
    return converge_points_time


# Find the convergence rate of reconstruction errors
def find_convergence_rate(errs):
    rates = []
    for i in range(1, len(errs)):
        rates.append(abs((errs[i] - errs[i-1])) / errs[i-1])
    rates = np.array(rates) * 100
    return rates


# Plot reconstruction error and convergence rate line 
# versus running time and number of iterations
def plot_error_time_iters(errors, running_times, converge_points, converge_points_time, 
                            converge_rate, color='b', rate_arrow_fix_pos=10**3, title=''):
    max_error = errors[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(range(1, 101), errors, c=color)

    k = [1.2, 2., 1.2]
    for i, cp in enumerate(converge_points):
        if cp == 0:
            continue
        axes[0].scatter(cp, errors[cp], s=100,
                c=[[1, 1, 1]], edgecolors='r')
        axes[0].annotate(f'T{i+1}', xy =(cp,
                    errors[cp]),
                    xytext =(cp-1.5, errors[cp]+rate_arrow_fix_pos*1.2),
                    arrowprops = dict(facecolor ='g', shrink = 0.05),)

        axes[0].annotate(f'{np.around(errors[cp], 2)}', xy=(cp, errors[cp]+rate_arrow_fix_pos*1.2+500), 
                            xytext=(cp, errors[cp]+rate_arrow_fix_pos*1.2+k[i]*rate_arrow_fix_pos),
                            arrowprops=dict(facecolor='black', arrowstyle='-', linestyle='--'))
        
    axes[0].annotate(f'{np.around(max_error, 2)}', xy=(0, max_error), xytext=(0+10, max_error-100),
                arrowprops=dict(facecolor='black', arrowstyle='-', linestyle='--'))

    axes[0].set_xlabel('Iterations', fontsize=14)
    axes[0].set_ylabel('Squared Frobenius', fontsize=14)
    axes[0].set_xticks([1]+list(range(10, 101, 10)))
    axes[0].grid()
    axes[0].set_title('(a)')

    axes[1].plot(running_times, errors, c=color)

    for i, cp in enumerate(converge_points):
        if cp == 0:
            continue
        axes[1].scatter(converge_points_time[i], errors[cp], s=100,
                    c=[[1, 1, 1]], edgecolors='r')
        axes[1].annotate(f'T{i+1}', xy =(converge_points_time[i],
                    errors[cp]),
                    xytext =(converge_points_time[i] - 0.4, errors[cp]+rate_arrow_fix_pos*1.2),
                    arrowprops = dict(facecolor ='g', shrink = 0.001),)

    axes[1].set_xlabel('Wall-clock Time (s)', fontsize=14)
    axes[1].set_ylabel('Squared Frobenius', fontsize=14)
    axes[1].grid()
    axes[1].set_title('(b)')

    axes[2].scatter(range(2, 101), converge_rate, s=10., c=color)
    axes[2].set_xlabel("Iterations", fontsize=14)
    axes[2].set_ylabel("Convergence Rate (%)", fontsize=14)
    axes[2].set_xticks([2]+list(range(10, 101, 10)))
    axes[2].grid()
    axes[2].set_title('(c)')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    

# Plot reconstruction error when restarting many times 
def plot_multi_repeats(errs_hist, name=''):
    errs_hist_var = np.var(errs_hist, axis=0)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    plots = []
    for i in range(len(errs_hist)):
        plts = ax.plot(range(1, 101), errs_hist[i], label=f'{name}_rep_{i+1}')
        plots.append(plts)

    plots_sum = plots[0] + plots[1] + plots[2] + plots[3] + plots[4]

    ax2 = ax.twinx()
    plt_var = ax2.plot(range(1, 101), errs_hist_var, color='black', linestyle='--', label='Variance')
    plots_sum = plots_sum + plt_var

    labels = [l.get_label() for l in plots_sum]
    ax.legend(plots_sum, labels, loc=0)
    ax.grid(True)

    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Squared Frobenius', fontsize=14)
    ax2.set_ylabel('Variance', fontsize=14)

    ax.set_xticks([1]+list(range(10, 101, 10)))
    plt.show()

    
# Running Task 1 
def task_1(A):
    # nmf_als with A and k = 20
    (W_als, H_als, errs_als, running_times_als) = nmf(A, 20, optFunc=nmf_als, maxiter=100, repetitions=1)
    converge_points_als = find_converge_points(errs_als)
    converge_points_time_als = find_converge_points_time(errs_als, running_times_als, converge_points_als)
    nmf_als_cr = find_convergence_rate(errs_als)
    plot_error_time_iters(errs_als, running_times_als, converge_points_als, converge_points_time_als, nmf_als_cr, color='purple', title='')


if __name__ == '__main__':
    # Load data
    ## Load the news data
    data = np.genfromtxt('news.csv', delimiter=',', skip_header=1)
    ## To read the terms, just read the first line of news.csv
    with open('news.csv') as f:
        header = f.readline()
        terms = [x.strip('"\n') for x in header.split(',')]
    