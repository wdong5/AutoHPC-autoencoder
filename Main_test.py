import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import tensorflow.python.framework.dtypes

from numpy import vstack
#from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from feature_reduction import *

import pandas as pd
import autokeras as ak
import timeit
import time
import pdb

from configuration import args
from prepare_data import generate_datasets, normalize
from Visualization_Bayesian import *

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import timeit
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def Input_reduction(X_matrix, y_matrix, reduction_ratio):
    x_data = np.array(X_matrix)
    X_batch = []
    Y_batch = []
    #for i in range(x_data.shape[0]):
    timesum_sparse_to_dense = []
    timesum_featurereduction = []
    for i in range(len(x_data)):
        # import pdb; pdb.set_trace()
        std_start = time.clock()
        x = np.array(x_data[i].todense())
        # import pdb; pdb.set_trace()
        std_end = time.clock()
        timesum_sparse_to_dense.append(std_end-std_start)
        X = normalize(x)
        # re_X = Shannon_reduction(X, reduction_ratio)
        # re_X = np.array(re_X)
        re_X = PCA_reduction(X, reduction_ratio)
        X_batch.append(re_X.flatten())
        Y_batch.append(y_matrix[i])
        red_end = time.clock()
        timesum_featurereduction.append(red_end - std_end)
    print("\tSparse to Dense Time:" + str(sum(timesum_sparse_to_dense)/len(timesum_sparse_to_dense)))
    print("\tPCA reduction Time:" + str(sum(timesum_featurereduction)/len(timesum_featurereduction)))
    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    return X_batch, Y_batch

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # get the dataset
    read_start = time.clock()
    X_train_sparse, Y_train, X_test_sparse, Y_test = generate_datasets()
    read_end = time.clock()
    read_timeconsume = read_end - read_start
    print("\tData Read Time:" + str([round(float(read_timeconsume), 5)]))
    x = 0.5
    x_train, y_train = Input_reduction(X_train_sparse, Y_train, x)
    x_test, y_test = Input_reduction(X_test_sparse, Y_test, x)
