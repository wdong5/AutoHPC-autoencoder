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
from manual_model import Manual_model
from prepare_data import generate_datasets, normalize
from Visualization_Bayesian import *

from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import timeit
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def csr2dense(X_train_sparse):
    x_data = np.array(X_train_sparse)
    X_batch = []
    for i in range(len(x_data)):
        x = np.array(x_data[i].todense())
        X_batch.append(x)
    X_batch = np.array(X_batch)
    return X_batch

def black_box_function(x):
    """Function with unknown internals we wish to maximize.
    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    # use encoder to reduce features of sparse matrix
    # if (args.benchmark=='CG' or args.benchmark=='AMG'):
    if (args.benchmark=='CG'):
        X_data = csr2dense(X_train_sparse)
        sizeof_X = X_data[0].shape[1]
        RED_FEATURE = int(x * sizeof_X)
        X_encoding = Embedding_matrix(X_data, encoding_dim = RED_FEATURE)
    # elif (args.benchmark=='MG' or args.benchmark=='Lagos_fine' or args.benchmark=='Lagos_coarse'):
    elif (args.benchmark=='MG' or args.benchmark=='Lagos_fine' or args.benchmark=='Lagos_coarse' or args.benchmark=='AMG'):
        X_data = np.array(X_train_sparse)
        sizeof_X = X_data.shape[1]
        RED_FEATURE = int(x * sizeof_X)
        X_encoding = Embedding_matrix(X_data, encoding_dim = RED_FEATURE)

    split_index = int(math.floor(len(X_encoding)* args.TRAIN_SET_RATIO / 100.0))
    assert (split_index >= 0 and split_index <= len(X_encoding))
    X_train = X_encoding[:split_index]
    X_test = X_encoding[split_index:]
    Y_train = np.array(Y_train_data[:split_index])
    Y_test = np.array(Y_train_data[split_index:])
    if (args.searchType=='autokeras'):
        print("******************Search model with Autokeras*******************************")
        ml_loss = Model_search(X_train, Y_train, X_test, Y_test, x)
    elif (args.searchType=='manualModel'):
        print("******************Search model manually with encoding*******************************")
        ml_loss = Manual_model(X_train, Y_train, X_test, Y_test, x)
    elif (args.searchType=='fullInput'):
        print("******************Search model manually with full input*******************************")
        X_full_train = X_data[:split_index]
        X_full_test = X_data[split_index:]
        ml_loss = Manual_model(X_full_train, Y_train, X_full_test, Y_test, x)

    return ml_loss#, initial_history, final_history

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # get the dataset
    read_start = time.clock()
    X_train_sparse, Y_train_data = generate_datasets()
    read_end = time.clock()
    read_timeconsume = read_end - read_start
    print("\tData Read Time:" + str([round(float(read_timeconsume), 5)]))

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds={'x': (0.01, 1)},
        random_state=20,
    )
    optimizer.maximize(
        init_points=20,
        n_iter=100,
    )

    f = open(os.path.join(args.save_bayesian_path, "Loss_log.txt"), "a")
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
        f.write(', '.join(str(i)+''+str(res))+'\n')
    print(optimizer.max)
    f.write(', '.join(str(optimizer.max))+'\n')
    f.close()
