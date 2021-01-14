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
from autoencoder import *
from autokeras_search import Model_search

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

def Input_dense(X_matrix, RED_FEATURE):
    x_data = np.array(X_matrix)
    X_batch = []
    for i in range(len(x_data)):
        x = np.array(x_data[i].todense())
        X_batch.append(x)
    X_batch = np.array(X_batch)
    sizeof_X = X_batch[0].shape[1]
    print(sizeof_X)
    # import pdb; pdb.set_trace()
    RED_FEATURE = int(RED_FEATURE * sizeof_X)
    X_batch = Embedding_matrix(X_batch, encoding_dim = RED_FEATURE)
    return X_batch

def black_box_function(x):
    """Function with unknown internals we wish to maximize.
    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # use encoder to reduce features of sparse matrix
    X_train= Input_dense(X_train_sparse, x)
    X_test= Input_dense(X_test_sparse, x)
    # import pdb; pdb.set_trace()
    ml_loss = Model_search(X_train, Y_train, X_test, Y_test, x)
    return ml_loss#, initial_history, final_history

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

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds={'x': (0.01, 1)},
        random_state=1,
    )
    optimizer.maximize(
        init_points=1,
        n_iter=0,
    )

    f = open(os.path.join(args.save_bayesian_path, "Loss_log.txt"), "a")
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        f.write(', '.join(str(i)+''+str(res))+'\n')
    print(optimizer.max)
    f.write(', '.join(str(optimizer.max))+'\n')
    f.close()
